/**
 * Microphone capture that produces a WAV file the backend can actually read.
 *
 * Why not MediaRecorder
 * ---------------------
 * MediaRecorder is the obvious choice and the wrong one here. Chrome gives you
 * `audio/webm;codecs=opus`, Safari gives `audio/mp4`, and the backend decodes
 * with libsndfile, which reads neither. The upload would be rejected with
 * "could not decode ... as audio", which looks like a backend bug and is not.
 *
 * So this captures raw float samples, downmixes and resamples to the 16 kHz
 * mono the model expects, and writes a WAV header itself. The output is
 * decodable by soundfile everywhere, and it is the same format the live
 * websocket streams, so a file test and a live test exercise identical audio.
 *
 * ScriptProcessorNode is deprecated in favour of AudioWorklet. It is used
 * anyway: AudioWorklet needs a separately served module file, which is
 * awkward under Next's bundler, and this runs for seconds at a time in a demo
 * rather than continuously in production.
 */

export const TARGET_SAMPLE_RATE = 16000;

export interface Recording {
  blob: Blob;
  seconds: number;
  peak: number;
}

/**
 * Linear resampling to the target rate.
 *
 * The browser will usually already give us 16 kHz because that is what
 * getUserMedia is asked for, but it is a hint, not a guarantee - some devices
 * ignore it and hand back 44.1 or 48 kHz. Uploading that unresampled is not
 * fatal (the backend resamples too) but it triples the upload for nothing.
 */
function resample(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return input;

  const ratio = from / to;
  const length = Math.floor(input.length / ratio);
  const output = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    const position = i * ratio;
    const index = Math.floor(position);
    const fraction = position - index;
    const a = input[index] ?? 0;
    const b = input[index + 1] ?? a;
    output[i] = a + (b - a) * fraction;
  }

  return output;
}

export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeText = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

export class MicRecorder {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private chunks: Float32Array[] = [];
  private sampleRate = TARGET_SAMPLE_RATE;

  /** Latest input level, for a meter. Updated on every audio callback. */
  level = 0;

  onLevel: ((level: number) => void) | null = null;

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: TARGET_SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true,
        // Left off deliberately. Automatic gain control compresses the quiet
        // parts of speech towards the loud parts, and the backend's
        // speech-presence check reads exactly that dynamic range to tell
        // speech from steady noise.
        autoGainControl: false,
      },
    });

    this.context = new AudioContext();
    this.sampleRate = this.context.sampleRate;
    this.chunks = [];

    this.source = this.context.createMediaStreamSource(this.stream);
    this.processor = this.context.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));

      let peak = 0;
      for (let i = 0; i < input.length; i++) {
        const magnitude = Math.abs(input[i]);
        if (magnitude > peak) peak = magnitude;
      }
      this.level = peak;
      this.onLevel?.(peak);
    };

    this.source.connect(this.processor);

    // ScriptProcessorNode only fires while it is connected to a destination.
    // Routing it to the speakers would echo the microphone back into the room,
    // so it goes to a zero gain node instead: connected, but silent.
    const silent = this.context.createGain();
    silent.gain.value = 0;
    this.processor.connect(silent);
    silent.connect(this.context.destination);
  }

  stop(): Recording | null {
    this.processor?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.context?.close().catch(() => {});

    const captured = this.chunks;
    const rate = this.sampleRate;

    this.processor = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.chunks = [];
    this.level = 0;

    if (captured.length === 0) return null;

    const total = captured.reduce((sum, chunk) => sum + chunk.length, 0);
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of captured) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    const resampled = resample(merged, rate, TARGET_SAMPLE_RATE);

    let peak = 0;
    for (let i = 0; i < resampled.length; i++) {
      const magnitude = Math.abs(resampled[i]);
      if (magnitude > peak) peak = magnitude;
    }

    return {
      blob: encodeWav(resampled, TARGET_SAMPLE_RATE),
      seconds: resampled.length / TARGET_SAMPLE_RATE,
      peak,
    };
  }

  get recording(): boolean {
    return this.context !== null;
  }
}

/**
 * The message a user needs when getUserMedia fails.
 *
 * "NotAllowedError" on its own tells them nothing; each of these has a
 * different fix and they are easy to confuse.
 */
export function microphoneError(error: unknown): string {
  const name = (error as { name?: string })?.name ?? "";

  if (name === "NotAllowedError" || name === "SecurityError") {
    return (
      "Microphone permission was denied. Click the padlock in the address " +
      "bar and allow the microphone, then try again."
    );
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No microphone was found. Plug one in, or check Windows sound settings.";
  }
  if (name === "NotReadableError") {
    return "The microphone is in use by another application. Close it and try again.";
  }

  return `Could not open the microphone: ${
    (error as Error)?.message ?? String(error)
  }`;
}
