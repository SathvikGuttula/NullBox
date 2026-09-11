```

======================================
  HEADLINE              full train set
======================================
  EER                            2.95%
  ROC-AUC                       0.9863
  normalised minDCF             0.1474

  at the EER threshold
    false alarm                  2.95%
    miss                         2.95%

======================================
  MISS RATE BY ATTACK   full train set
======================================
  A07 TTS                        0.04%
  A08 TTS                        0.00%
  A09 TTS                        0.00%
  A10 TTS                        9.32%
  A11 TTS                        0.08%
  A12 TTS                        0.00%
  A13 TTS_VC                     0.00%
  A14 TTS_VC                     0.00%
  A15 TTS_VC                     0.20%
  A16 TTS                        0.14%
  A17 VC                         7.69%
  A18 VC                        18.09%
  A19 VC                         2.79%
  ------------------------------------
  bonafide (FA)                  2.95%

  BY SYNTHESIS FAMILY  (mean miss rate)
    TTS                          1.37%
    TTS_VC                       0.07%
    VC                           9.52%

  A05/A06 are the ONLY voice-conversion attacks in ASVspoof 2019 LA
  train and dev; A17-A19 are the voice-conversion attacks in eval.
  Holding A05/A06 out therefore removes the entire VC family from
  training, and the VC row above is where that shows up.

======================================

```