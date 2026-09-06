
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.call import Call


async def create_call(
    db: AsyncSession,
    caller_name: str | None = None,
) -> Call:
    call = Call(
        id=uuid.uuid4(),
        caller_name=caller_name,
        status="active",
        risk_score=0.0,
    )

    db.add(call)
    await db.commit()
    await db.refresh(call)

    return call