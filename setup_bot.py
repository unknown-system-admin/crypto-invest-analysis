#!/usr/bin/env python3
"""Discord Bot Setup Script"""
import os
import sys
import yaml

print("=" * 50)
print("Discord Bot Setup")
print("=" * 50)
print()
print("Before running this script:")
print("1. Go to https://discord.com/developers/applications")
print("2. Click 'New Application' and create a bot")
print("3. Copy the Bot Token")
print("4. Go to OAuth2 > URL Generator")
print("5. Select 'bot' scope")
print("6. Select 'Send Messages' permission")
print("7. Copy the generated URL and open it to invite the bot")
print("8. Get the Channel ID (right-click channel > Copy ID)")
print()
bot_token = input("Enter Bot Token: ").strip()
channel_id = input("Enter Channel ID: ").strip()

if not bot_token or not channel_id:
    print("Error: Bot Token and Channel ID are required")
    sys.exit(1)

config_path = "monitor/config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

config["discord"]["bot_token"] = bot_token
config["discord"]["channel_id"] = channel_id

with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print()
print("Bot configuration saved!")
print()
print("Testing bot connection...")

import discord

intents = discord.Intents.default()
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Bot connected as {client.user}")
    channel = client.get_channel(int(channel_id))
    if channel:
        await channel.send("🤖 Bot connected successfully!")
        print("Test message sent!")
    else:
        print(f"Error: Channel {channel_id} not found")
    await client.close()

client.run(bot_token)
