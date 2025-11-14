from core.bot import Bot


if __name__ == "__main__":
    bot = Bot()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user (KeyboardInterrupt).")
