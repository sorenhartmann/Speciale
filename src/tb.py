import webbrowser
from pathlib import Path
from threading import Timer

from tensorboard import program

# tracking_address = log_path # the path of your log file.

if __name__ == "__main__":

    events_files = list(Path(__file__).parents[1].glob("**/events.out.tfevents*"))
    
    choices = {i : { "name":x.stem, "dir" : str(x)} for i, x in enumerate(set(x.parents[2] for x in events_files))}

    for i, v in choices.items():
        print(f"Press [{i}] for {v['name']}")
    
    choice = int(input())

    port = 6006

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--port', str(port), '--logdir', choices[choice]["dir"]])

    url = tb.main()