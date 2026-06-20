"""
Quick test: fetch all messages from support channel and print user + content.
"""

import asyncio
import discord
from shared.config import DISCORD_BOT_TOKEN, SUPPORT_CHANNEL_ID

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    channel = client.get_channel(SUPPORT_CHANNEL_ID)
    if not channel:
        print(f"Channel {SUPPORT_CHANNEL_ID} not found")
        await client.close()
        return

    print(f"Fetching history from #{channel.name}...\n")
    count = 0
    async for msg in channel.history(limit=None, oldest_first=True):
        print(f"[{msg.created_at}] {msg.author.id} ({msg.author.name}): {msg.content[:100]}")
        count += 1

    print(f"\nTotal: {count} messages")
    await client.close()


asyncio.run(client.start(DISCORD_BOT_TOKEN))