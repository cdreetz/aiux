# config/logging_config.py
import logging
import os
import json

class ConversationLogFormatter(logging.Formatter):
    def format(self, record):
        # Check if the message is a dictionary and format it accordingly
        if isinstance(record.msg, dict):
            if 'role' in record.msg and 'content' in record.msg:
                content = record.msg["content"].replace('\n', ' ').replace('"', '\\"')
                formatted_message = f'{{"role": "{record.msg["role"]}", "content": "{content}"}}'
            elif 'context' in record.msg:
                context = json.dumps(record.msg["context"]).replace('\n', ' ').replace('"', '\\"')
                formatted_message = f'{{"context": {context}}}'
            else:
                # Default formatting for other types of messages
                formatted_message = json.dumps(record.msg)
            return f'{self.formatTime(record, self.datefmt)} :: INFO :: {formatted_message}'
        return super().format(record)

def setup_logging(filename):
    # Ensure the logging directory exists
    log_directory = os.path.dirname(filename)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Create a new logger
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers

    # Define the format for the log messages
    log_format = '%(asctime)s :: %(levelname)s :: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # Add a file handler with a custom formatter
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(ConversationLogFormatter(fmt=log_format, datefmt=date_format))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
