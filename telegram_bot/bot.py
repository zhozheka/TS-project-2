#! /usr/bin/python3
# -*- coding: utf-8 -*-
import config
import telebot


token = ':'.join(config.token[::-1].split(':')[::-1])
bot = telebot.TeleBot(token)

@bot.message_handler(content_types=["text"])
def repeat_all_messages(message):
    bot.send_message(message.chat.id, message.text)

if __name__ == '__main__':
     bot.polling(none_stop=True)
