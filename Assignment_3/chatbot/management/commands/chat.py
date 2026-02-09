from django.core.management.base import BaseCommand
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


class Command(BaseCommand):
    help = "Start a terminal-based chatbot using ChatterBot"

    def handle(self, *args, **kwargs):
        # Initialize chatbot
        chatbot = ChatBot(
            'TerminalBot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            database_uri='sqlite:///db.sqlite3'
        )

        # Train chatbot with English corpus
        trainer = ChatterBotCorpusTrainer(chatbot)
        trainer.train('chatterbot.corpus.english')

        self.stdout.write(self.style.SUCCESS(
            "Chatbot is ready! Type 'exit' to quit.\n"
        ))

        # Chat loop
        while True:
            user_input = input("user: ")

            if user_input.lower() in ['exit', 'quit']:
                self.stdout.write("bot: Goodbye!")
                break

            response = chatbot.get_response(user_input)
            self.stdout.write(f"bot: {response}")
