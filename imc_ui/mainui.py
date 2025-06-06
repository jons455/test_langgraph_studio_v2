import tkinter as tk
from tkinter import scrolledtext
from tkinter import Event
from typing import Optional
import threading
import re

from imc_chatgpt.chatgpt_service import ChatGptService
from imc_chatgpt.context_handler import ContextHandler
from imc_norm.product_number_check_service import ProductNumberCheckService


class ChatUI:
    VALID_COLORS = ("red", "orange", "green")
    #195d86 light blue
    #191939 darker blue
    #33334b greyish blue
    #00d7a0 lighish green
    #66667e grey

    def __init__(self, root: tk.Tk) -> None:
        self.context_handler = None
        self.norm_service = None
        self.chatgpt_service = None

        self.root: tk.Tk = root
        self.root.title("Chat UI")
        self.root.configure(bg="#191939")  # Darker blue background

        # Configure grid for resizing
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.status_frame: tk.Frame = tk.Frame(root, bg="#191939")
        self.status_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        self.chatgpt_status_led: tk.Canvas = tk.Canvas(self.status_frame, width=20, height=20, highlightthickness=0,
                                                       bg="#191939")
        self.chatgpt_status_led.pack(side=tk.LEFT, padx=5)
        self.chatgpt_led = self.chatgpt_status_led.create_oval(2, 2, 18, 18, fill="#66667e")  # Gray by default
        self.chatgpt_label: tk.Label = tk.Label(self.status_frame, text="ChatGPT API", fg="white", bg="#191939")
        self.chatgpt_label.pack(side=tk.LEFT, padx=(0, 10))

        self.norm_status_led: tk.Canvas = tk.Canvas(self.status_frame, width=20, height=20, highlightthickness=0,
                                                    bg="#191939")
        self.norm_status_led.pack(side=tk.LEFT, padx=5)
        self.norm_led = self.norm_status_led.create_oval(2, 2, 18, 18, fill="#66667e")
        self.norm_label: tk.Label = tk.Label(self.status_frame, text="Norm API", fg="white", bg="#191939")
        self.norm_label.pack(side=tk.LEFT, padx=(0, 10))

        self.chat_area: scrolledtext.ScrolledText = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled',
                                                                              bg="#33334b", fg="white")
        self.chat_area.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.entry_frame: tk.Frame = tk.Frame(root, bg="#191939")
        self.entry_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.message_entry: tk.Text = tk.Text(self.entry_frame, height=3, width=50, bg="#195d86", fg="white",
                                              insertbackground="white")
        self.message_entry.pack(side=tk.LEFT, padx=(0, 10), expand=True, fill='x')
        self.message_entry.bind("<Shift-Return>", self._on_send)

        self.send_button: tk.Button = tk.Button(self.entry_frame, text="Send", command=self._on_send, bg="#00d7a0",
                                                fg="black")
        self.send_button.pack(side=tk.RIGHT)

    def set_chatgpt_service(self, chatgpt_service: ChatGptService) -> None:
        self.chatgpt_service = chatgpt_service
        if self.chatgpt_service:
            self.set_chatgpt_status("green")
            self.context_handler = ContextHandler(chatgpt_service)
        else:
            print("ChatGPT was attempted to set but did not result in a valid value.")
            self.set_chatgpt_status("red")

    def set_norm_service(self, norm_service: ProductNumberCheckService) -> None:
        self.norm_service = norm_service
        if self.norm_service:
            self.set_norm_status("green")
        else:
            print("Norm service was attempted to set but did not result in a valid value.")
            self.set_norm_status("red")

    def add_user_message(self, message: str, silent: bool = False) -> None:
        if not isinstance(message, str) or not message.strip():
            return

        if not silent:
            self._add_message("You", message, "white")

        if message.startswith("NORM"):
            remaining_message = message[4:]
            if self.norm_service:
                response = self.norm_service.validate_product_number(remaining_message)
                print("Response from Norm service:", response)
                self.add_bot_message(str(response))
            else:
                print("Norm service is not set.")
                self.set_norm_status("red")
            return

        self.set_chatgpt_status("orange")
        threading.Thread(target=self._blocking_send_message_to_services, args=(message,)).start()

    def _blocking_send_message_to_services(self, message: str) -> None:
        if self.chatgpt_service:
            contextedMessage = dict()
            contextedMessage["context"] = self.context_handler.context
            contextedMessage["input"] = message
            response = self.context_handler.send_message_in_context(message)
            self.add_bot_message(response)
            self.set_chatgpt_status("green")
        else:
            print("ChatGPT service is not set.")
            self.set_chatgpt_status("red")

    def add_bot_message(self, message: str, silent: bool = True) -> None:
        if not isinstance(message, str) or not message.strip():
            return

        self._add_message("Bot", message, "#00d7a0")

    def _add_message(self, sender: str, message: str, color: str) -> None:
        if not sender or not message:
            return
        self.chat_area.config(state='normal')
        try:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n", (color,))
            self.chat_area.tag_config(color, foreground=color)
        except Exception:
            self.chat_area.insert(tk.END, f"{sender}: {message}\n")
        self.chat_area.insert(tk.END, "------------------------------\n", ("separator",))
        self.chat_area.tag_config("separator", foreground="gray")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def _on_send(self, event: Optional[Event] = None) -> None:
        if event:
            try:
                event.preventDefault = True
            except AttributeError:
                pass
        message: str = self.message_entry.get("1.0", tk.END).strip()
        if message:
            self.add_user_message(message)
            self.message_entry.delete("1.0", tk.END)

    def set_chatgpt_status(self, color: str) -> None:
        if color in self.VALID_COLORS:
            self.chatgpt_status_led.itemconfig(self.chatgpt_led, fill=color)
        else:
            print("WARNING: Invalid color for ChatGPT status. Defaulting to gray.")
            self.chatgpt_status_led.itemconfig(self.chatgpt_led, fill="gray")

    def set_norm_status(self, color: str) -> None:
        if color in self.VALID_COLORS:
            self.norm_status_led.itemconfig(self.norm_led, fill=color)
        else:
            self.norm_status_led.itemconfig(self.norm_led, fill="gray")
