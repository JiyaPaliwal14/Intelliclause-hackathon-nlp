# create_tables.py

import asyncio
from app.database import engine
from app.models_db import Base

async def create_all_tables():
    """Creates all database tables."""
    async with engine.begin() as conn:
        print("Dropping all tables (if they exist)...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Creating all tables...")
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created successfully.")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(create_all_tables())