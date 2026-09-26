import customtkinter as ctk
import time

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PosturaNodeApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PosturaNode")
        self.geometry("320x420")
        self.resizable(False, False)
        self.attributes("-topmost", True)

        # Basic UI Setup
        self.status_indicator = ctk.CTkLabel(self, text="GOOD POSTURE", font=ctk.CTkFont(size=16, weight="bold"), text_color="#4CAF50")
        self.status_indicator.pack(pady=20)

        self.score_value = ctk.CTkLabel(self, text="98", font=ctk.CTkFont(size=56, weight="bold"))
        self.score_value.pack(pady=10)

        # Slouch Simulation (Since Backend isn't ready)
        self.is_slouching = False
        self.slouch_start_time = None
        
        self.simulate_btn = ctk.CTkButton(self, text="Simulate: Start Slouching", command=self.toggle_simulation, fg_color="#E67E22", hover_color="#D35400")
        self.simulate_btn.pack(pady=10, padx=20, fill="x")

        # Demo Mode Toggle
        self.demo_switch = ctk.CTkSwitch(self, text="Demo Mode (Instant Alert)", command=self.trigger_demo_alert)
        self.demo_switch.pack(pady=15)

        self.alert_window = None

        # Start the background checker loop
        self.check_intervention_loop()

    def toggle_simulation(self):
        """Simulates the backend detecting a slouch so you can test the timer."""
        self.is_slouching = not self.is_slouching
        
        if self.is_slouching:
            self.simulate_btn.configure(text="Simulate: Sit Up Straight", fg_color="#27AE60", hover_color="#2ECC71")
            self.status_indicator.configure(text="SLOUCHING DETECTED", text_color="#F44336")
            self.score_value.configure(text="42") # Fake low score
            self.slouch_start_time = time.time() # Start the timer
        else:
            self.simulate_btn.configure(text="Simulate: Start Slouching", fg_color="#E67E22", hover_color="#D35400")
            self.status_indicator.configure(text="GOOD POSTURE", text_color="#4CAF50")
            self.score_value.configure(text="98") # Fake high score
            self.slouch_start_time = None # Reset the timer

    def trigger_demo_alert(self):
        """Instantly forces the intervention alert for presentation purposes."""
        if self.demo_switch.get():
            self.show_intervention_alert()

    def check_intervention_loop(self):
        """Runs every 1 second in the background to check the 45-second rule."""
        if self.is_slouching and self.slouch_start_time:
            elapsed_time = time.time() - self.slouch_start_time
            
            # If they have been slouching for 45 seconds (we use 5 seconds here just for quick testing!)
            if elapsed_time >= 5: 
                self.show_intervention_alert()
                self.slouch_start_time = None # Stop triggering it repeatedly

        # Schedule this function to run again in 1000 milliseconds (1 second)
        self.after(1000, self.check_intervention_loop)

    def show_intervention_alert(self):
        """Displays the screen-dimming overlay forcing a break."""
        if self.alert_window is not None and self.alert_window.winfo_exists():
            return # Don't open multiple windows

        self.alert_window = ctk.CTkToplevel(self)
        self.alert_window.title("Posture Alert")
        self.alert_window.geometry("500x300")
        self.alert_window.attributes("-topmost", True)
        
        # Center the alert on the screen
        self.alert_window.update_idletasks()
        x = (self.alert_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.alert_window.winfo_screenheight() // 2) - (300 // 2)
        self.alert_window.geometry(f"+{x}+{y}")

        title = ctk.CTkLabel(self.alert_window, text="Time for a Stretch Break! 🧘", font=ctk.CTkFont(size=24, weight="bold"))
        title.pack(pady=(50, 20))

        msg = ctk.CTkLabel(self.alert_window, text="You have been slouching for 45 seconds.\nSit up straight to dismiss this alert.", font=ctk.CTkFont(size=14))
        msg.pack(pady=10)

        dismiss_btn = ctk.CTkButton(self.alert_window, text="I fixed my posture", command=self.alert_window.destroy)
        dismiss_btn.pack(pady=20)

if __name__ == "__main__":
    app = PosturaNodeApp()
    app.mainloop()
