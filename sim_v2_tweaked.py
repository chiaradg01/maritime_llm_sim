## Imports ##
import matplotlib.pyplot as plt
from openai import OpenAI
import random
import time
import os

## Setup Results folder ##
os.makedirs("results_v2", exist_ok=True)

## The System ##
class Ship:
    def __init__(self, name, x, y, color):
        self.name = name
        self.x = x
        self.y = y
        self.color = color
        self.history = [(x,y)] # keep track of path
    
    def move(self, direction, grid_size):
        if direction == "North" and self.y < grid_size - 1:
            self.y += 1
        elif direction == "South" and self.y > 0:
            self.y -= 1
        elif direction == "East" and self.x < grid_size - 1:
            self.x += 1
        elif direction == "West" and self.x > 0:
            self.x -= 1
        elif direction == "Wait":
            pass

        self.history.append((self.x, self.y))

class MaritimeSim:
    def __init__(self, grid_size = 10):
        self.grid_size = grid_size
        self.defender = Ship(name="Coast Guard", x=2, y=2, color="blue")
        self.intruder = Ship(name="Smuggler", x=8, y=8, color="red")
    
    def render(self, run_id, step):
        """
        Draws grid and saves it to results folder.
        """
        plt.figure(figsize=(6, 6))
        plt.grid(True)
        plt.xlim(-1, self.grid_size)
        plt.ylim(-1, self.grid_size)

        # Draw ships
        plt.scatter(self.defender.x, self.defender.y, c=self.defender.color, s=200, label=self.defender.name, marker="s")
        plt.scatter(self.intruder.x, self.intruder.y, c=self.intruder.color, s=200, label=self.intruder.name, marker ="x")

        # Draw paths
        def_x, def_y = zip(*self.defender.history)
        int_x, int_y = zip(*self.intruder.history)
        plt.plot(def_x, def_y, c=self.defender.color, linestyle="--", alpha=0.5)
        plt.plot(int_x, int_y, c=self.intruder.color, linestyle="--", alpha=0.5)

        plt.title(f"Run {run_id} | Maritime Simulation - Step {step}")
        plt.legend(loc="upper left")

        # Save dynamically with run number and step
        plt.savefig(f"results_v2/run_{run_id}_step_{step}.png")
        plt.close()

## LLM (GenAI) ##
def get_coast_guard_move(defender, intruder, client):
    """
    Asks Ollama (Mistral) to decide the next move.
    """
    system_prompt = """You are a highly logical Coast Guard AI intercepting a smuggler.
    Compare your coodinates to the smuggler's coordinates and moves towards them.

    CRITICAL ALGORITHMIC RULES:
    - If smuggler Y is less than your Y, just must move South.
    - If smuggler Y is greater than your Y, you must move North.
    - If smuggler X is less than your X, you must move West.
    - If smuggler X is greater than your X, you must move East.
    - CONFLICT RESOLUTION: If the smuggler is diagonal to you (e.g. both North and East), you must pick EXACTLY ONE direction to move this turn. Do not combine words.
    You must output EXACTLY ONE WORD indicating your move: North, South, East, West, or Wait.
    Do NOT explain your reasoning. Do NOT output any punctuation."""
    
    user_prompt = f"You are at X:{defender.x} Y:{defender.y}. The smuggler is at X:{intruder.x} Y:{intruder.y}. What is your move?"

    try:
        response = client.chat.completions.create(
            model = "mistral:7b-instruct", # ollama model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        raw_answer = response.choices[0].message.content.strip()

        # Clean up answer
        for move in ["North", "South", "East", "West", "Wait"]:
            if move.lower() in raw_answer.lower():
                return raw_answer, move
        return raw_answer, "Wait"
    
    except Exception as e:
        return f"Error: {e}", "Wait"

## Run Simulation & Logging ##
if __name__ == "__main__":
    print("Starting Batch simulation setup...")
    
    # Connect to local Ollama server
    client = OpenAI(base_url= "http://localhost:11434/v1", api_key= "ollama")

    total_runs = 10
    steps_per_run = 15

    # Successful interception tracker (counter)
    successful_interceptions = 0 # initialization

    # Open log file
    with open("results_v2/simulation_logs.txt", "w") as log_file:
        log_file.write("MARITIME LLM SIMULATION LOGS \n")
        log_file.write("================================\n\n")


        # Run for 6 turns

        for run_id in range(1, total_runs + 1):
            run_header = f"--- Starting Run {run_id} ---\n"
            print(run_header.strip())
            log_file.write(run_header)

            sim = MaritimeSim(grid_size=10)
            sim.render(run_id=run_id, step=0)

            for current_step in range(1, steps_per_run + 1):
                step_header = f"\nRun {run_id} | Turn {current_step}\n"
                print(step_header.strip())
                log_file.write(step_header)

                # Defender (Coast Guard) moves (LLM)
                raw_ans, llm_move = get_coast_guard_move(sim.defender, sim.intruder, client)
                sim.defender.move(llm_move, sim.grid_size)

                cg_log = f"Coast Guard (LLM) answered '{raw_ans}' -> Actually moves: {llm_move}\n"
                print(cg_log.strip())
                log_file.write(cg_log)

                # Intruder moves (Dumb logic)
                intruder_move = random.choices(["South", "West", "Wait"], weights = [45, 45, 10])[0]
                sim.intruder.move(intruder_move, sim.grid_size)

                sm_log = f"Smuggler moves: {intruder_move}\n"
                print(sm_log.strip())
                log_file.write(sm_log)

                # Render frame
                sim.render(run_id=run_id, step=current_step)

                # Check Interception
                if sim.defender.x == sim.intruder.x and sim.defender.y == sim.intruder.y:
                    successful_interceptions += 1
                    success_msg = f"\n*** INTERCEPTION SUCCESsFUL! Coast Guard caught the smuggler at Turn {current_step}! ***\n"
                    print(success_msg.strip())
                    log_file.write(success_msg)
                    break
            run_footer = f"\n --- END OF RUN {run_id} --- \n\n"        
            print(run_footer)
            log_file.write(run_footer)    
        
        # --- Final V2 Summary ---
        summary_msg = f"\n================================\n"
        summary_msg += f"FINAL V2 RESULTS: {successful_interceptions}/{total_runs} Interceptions\n"
        summary_msg += f"SUCCESS RATE V2: {(successful_interceptions/total_runs)*100}%\n"
        summary_msg += f"================================\n"

        print(summary_msg)
        log_file.write(summary_msg)
    
    print("Batch Simulation complete. Check folder for logs and images.")


