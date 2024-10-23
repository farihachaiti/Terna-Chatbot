import asyncio
import chainlit as cl
from pdf_processor import process_all_pdfs

@cl.on_message
def handle_message(message):
    # Directly access the text property of the Message object
    command = message.text  # Use message.text to access the command
    if command is not None and command.strip() == "process_all_pdfs":
        process_all_pdfs("AllPdfs")  # Specify the folder name correctly
        cl.Message("All PDFs processed successfully!", "success")
    else:
        cl.Message("Unknown command. Please use 'process_all_pdfs' to process PDFs.", "error")

async def main():
    await cl.start()  # Start the Chainlit app

if __name__ == "__main__":
    asyncio.run(main())
