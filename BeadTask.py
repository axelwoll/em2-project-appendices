import psychopy
#psychopy.useVersion('2023.1.3')

from psychopy import visual, core, event, gui
import random
import math
import numpy as np
import csv

# Create window
win = visual.Window(fullscr=True, color="grey", waitBlanking=True)
text_color = 'black'
default_font = 'DejaVu Sans'

# -------------------
# Collect subject number
# -------------------
subject_number = ""
text_stim = visual.TextStim(win, text="Enter subject number: ", pos=(0, 0.2), color = text_color, font = default_font)
response_stim = visual.TextStim(win, text="", pos=(0, -0.2), color = text_color, font = default_font)

while True:
    text_stim.draw()
    response_stim.text = subject_number
    response_stim.draw()
    win.flip()

    keys = event.waitKeys()
    if 'return' in keys:
        break
    elif 'backspace' in keys:
        subject_number = subject_number[:-1]
    elif len(keys[0]) == 1:
        subject_number += keys[0]

RATIO_MAP = {
    60: (60, 40),
    90: (90, 10)
}

SEQUENCES = {
    60: [ # 16
        [1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1],
        [0,1,1,1,0,1,1,0],
        [0,1,1,0,1,1,1,0],
        [1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,1],
        [1,0,1,1,1,1,0,1],
        [1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1],
        [1,1,1,1,1,0,1,1],
        [1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,0,1,0,0],
        [0,0,1,0,1,1,1,1],
        [1,1,0,1,1,1,0,0],
        [0,0,1,1,1,0,1,1]
    ],
    90: [ # (11 + 5)
        [1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1],
        [0,1,1,1,0,1,1,0],
        [0,1,1,0,1,1,1,0],
        [1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,1],
        [1,0,1,1,1,1,0,1],
        [1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1],
        [1,1,1,1,1,0,1,1],
        [1,1,0,1,1,1,1,1],
        [1,1,1,1,1,1,1,1], 
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1]
    ]
}

BLOCK_ORDER = [60, 90, 60, 90] 

# -------------------
# Other
# -------------------
def destretch_stimuli(stimuli, win):
    """
    Corrects horizontal stretching for stimuli in a 'norm' coordinate system.
    Scales widths and x-positions so shapes look proportionally correct.

    Parameters
    ----------
    stimuli : list
        List (or iterable) of PsychoPy stimuli (e.g., Rect, Circle, TextStim, etc.)
    win : psychopy.visual.Window
        The PsychoPy window (used to get aspect ratio)
    """
    aspect = win.size[0] / win.size[1]  # width / height

    for stim in stimuli:
        # Adjust X position
        if hasattr(stim, 'pos'):
            stim.pos = (stim.pos[0] / aspect, stim.pos[1])
        # Adjust width (or size tuple)
        if hasattr(stim, 'width') and hasattr(stim, 'height'):
            stim.width /= aspect
        elif hasattr(stim, 'size'):
            stim.size = (stim.size[0] / aspect, stim.size[1])


# -------------------
# Experiment class
# -------------------
class BeadsTask:
    def __init__(self, win, subject_id):
        self.win = win
        self.subject_id = subject_id
        self.results = []

        # Pre-create and cache visual stimuli to avoid repeated creation
        self.blue_box = visual.Rect(win, width=0.2, height=0.2,
                                    fillColor='blue', lineColor='blue')
        self.green_box = visual.Rect(win, width=0.2, height=0.2,
                                     fillColor='green', lineColor='green')
        
        # Pre-create and cache visual stimuli to avoid repeated creation
        self.slider = visual.Slider(
            win, 
            ticks=[0, 0.5, 1], 
            granularity=0,
            labels = ['', '', ''],
            style=['rating'], 
            size=(0.75, 0.1),
            pos=(0, -0.4),
            color = text_color,
            fillColor= text_color,  
            lineColor= text_color,
            font = 'DejaVu Sans'
        )
        
        # Pre-create mouse
        self.mouse = event.Mouse(win=self.win)
        
        # hide default marker
        self.slider.marker.color = None
        self.slider.marker.opacity = 0
        
        # Pre-calculate and cache common values
        self.weights = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        self.bead_radius = 0.07
        self.bead_spacing = 0.15
        self.total_beads = 8
        
        # Pre-create reusable visual elements
        self._create_reusable_stimuli()

        # Generate all trials automatically
        self.trials = self.generate_trials()
        self.practice_trials = self.generate_practice_trials()

    def _create_reusable_stimuli(self):
        """Pre-create reusable visual stimuli to avoid repeated creation during trials
        
           Ideally, all stimulus-objects which are to be drawn on screen more than once, could
           be CREATED with this method. The methods responsible for drawing, or gathering drawable
           objects in a list, to be drawn, will then only need to modify appropriate attributes
           of the pre-created objects.
           
           Also to ensure nice aspect-ratios (relating to width/height of objects relative to window size
           all precreated stimuli is also 'destretched' in this method, rather than every time they are drawn.
        
        """
   
        # Pre-create bead circles for different colors (specifically, those used in the draw-animations, and the visual record) and destretch
        self.bead_circles = {
            'blue': visual.Circle(self.win, radius=self.bead_radius, fillColor='blue', 
                                 lineColor='blue', pos=(0, 0)),
            'green': visual.Circle(self.win, radius=self.bead_radius, fillColor='green', 
                                  lineColor='green', pos=(0, 0)),
            'empty': visual.Circle(self.win, radius=self.bead_radius, fillColor='white', 
                                  lineColor='white', pos=(0, 0))
        }
        
        destretch_stimuli(self.bead_circles.values(), self.win)
        
        # Pre-create common text stimuli
        self.prior_text = visual.TextStim(
            self.win,
            text="Before seeing any beads,\nestimate the probability of the Hidden Box \n by clicking on the slider",
            color=text_color, pos=(0, 0.1), height=0.08
        )

        self.est_prob_text = visual.TextStim(
            self.win,
            text="Estimate the probability of the Hidden Box given the bead sample",
            color=text_color, pos = (0, 0.1), height=0.08
        )
        
        self.final_choice_text1 = visual.TextStim(
            self.win, 
            text="Which box was the Hidden Box? \n Press left arrow for the left box, right arrow for the right box",
            color=text_color, pos=(0, 0.1), height=0.08
        )
        
        destretch_stimuli([self.prior_text,self.est_prob_text, self.final_choice_text1], self.win)
        
        # Pre-create the stimulus-objects that make up the two boxes
    
        # Define parameters relating to box positions and dimensions, margins, bead size and number of beads.
        self.left_box_pos, self.right_box_pos = (-1.0, -0.4), (1.0, -0.4)
        self.n_rows, self.n_cols = 10, 10
        self.circle_radius = 0.02
        self.grid_width = self.n_cols * self.circle_radius * 2.5
        self.grid_height = self.n_rows * self.circle_radius * 2.5
        self.grid_margin = 0.02
        self.max_beads = self.n_rows * self.n_cols  # 100 beads per box
        
        # Make two lists, holding the position of each of the 100 + 100 beads making up the two boxes
        self.green_bead_positions = []
        self.blue_bead_positions = []

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                x_green = self.left_box_pos[0] - (self.grid_width / 2 - self.grid_margin) + j * self.circle_radius * 2.5
                y_green = self.left_box_pos[1] + (self.grid_height / 2 - self.grid_margin) - i * self.circle_radius * 2.5
                self.green_bead_positions.append((x_green, y_green))

                x_blue = self.right_box_pos[0] - (self.grid_width / 2 - self.grid_margin) + j * self.circle_radius * 2.5
                y_blue = self.right_box_pos[1] + (self.grid_height / 2 - self.grid_margin) - i * self.circle_radius * 2.5
                self.blue_bead_positions.append((x_blue, y_blue))
        
        # Pre-create beads for green box (positions are predefined, while fillColor will be decided in the draw_boxes method)
        self.green_beads = []
        for i in range(self.max_beads):
            circ = visual.Circle(
                self.win,
                radius=self.circle_radius,
                fillColor='white',  # default color, will update per trial
                lineColor='white',
                pos=self.green_bead_positions[i]
            )
            self.green_beads.append(circ)

        # Pre-create beads for blue box (positions are predefined, while fillColor will be decided in the draw_boxes method)
        self.blue_beads = []
        for i in range(self.max_beads):
            circ = visual.Circle(
                self.win,
                radius=self.circle_radius,
                fillColor='white',
                lineColor='white',
                pos=self.blue_bead_positions[i]
            )
            self.blue_beads.append(circ)

        # Pre-create frames
        self.green_frame = visual.Rect(
            self.win,
            width= self.grid_width + self.grid_margin,
            height= self.grid_height + self.grid_margin,
            fillColor='dimgrey',
            lineColor='darkgreen',
            lineWidth=12,
            pos=self.left_box_pos  # default left box position
        )

        self.blue_frame = visual.Rect(
            self.win,
            width= self.grid_width + self.grid_margin,
            height= self.grid_height + self.grid_margin,
            fillColor='dimgrey',
            lineColor='darkblue',
            lineWidth=12,
            pos=self.right_box_pos # default right box position
        )
        
        self.green_frame.color_id = 'green'
        self.blue_frame.color_id = 'blue'
        
        # Apply destretch ONCE to the pre-created objects making up the two boxes
        destretch_stimuli(self.green_beads + self.blue_beads + [self.green_frame, self.blue_frame], self.win)
        
        # Pre-create the 'Green Box' / 'Blue Box' labels
        self.left_label = visual.TextStim(
                self.win, 
                text="Green box", 
                color='darkgreen',
                pos=(self.left_box_pos[0], self.left_box_pos[1] + (self.green_frame.height / 2) + 0.06), 
                height=0.08
                )
                
        self.right_label = visual.TextStim(
                self.win, 
                text="Blue box", 
                color='darkblue',
                pos=(self.right_box_pos[0], self.right_box_pos[1] + (self.blue_frame.height / 2) + 0.06), 
                height=0.08
                )
        
        # Ratio labels (initialized empty, update text per trial)
        self.green_ratio_text_top = visual.TextStim(self.win, text="", color=text_color, height=0.08)
        self.green_ratio_text_bottom = visual.TextStim(self.win, text="", color=text_color, height=0.08)
        self.blue_ratio_text_top = visual.TextStim(self.win, text="", color=text_color, height=0.08)
        self.blue_ratio_text_bottom = visual.TextStim(self.win, text="", color=text_color, height=0.08)
        
        # Destretch the left/right-labels and the ratio labels
        destretch_stimuli([self.left_label, self.right_label,
                       self.green_ratio_text_top, self.green_ratio_text_bottom,
                       self.blue_ratio_text_top, self.blue_ratio_text_bottom], self.win)
        
        
        # Pre-create objects related to the slider - the marker bar (yellow), and the dynamic labels
        
        # Slider marker
        self.marker_bar = visual.Rect(
            win=self.win,
            width=0.01,
            height=0.15,
            fillColor='yellow',
            lineColor='black',
            lineWidth=2,
            pos=(0, -0.4)  # initial position
        )

        # Slider end labels (dynamic text, reuse)
        self.slider_left_label = visual.TextStim(
            win=self.win,
            text='0%',           # placeholder
            pos=(-0.5, -0.55),   # placeholder
            color='black',
            height=0.06
        )

        self.slider_right_label = visual.TextStim(
            win=self.win,
            text='100%',
            pos=(0.5, -0.55),    # placeholder
            color='black',
            height=0.06
        )
        
        # Pre-create the question-mark box (for the animaton)
        self.question_box = visual.Rect(
                            self.win, 
                            width=self.green_frame.width, # Same size as the 'green' and 'blue box'
                            height=self.green_frame.height, # Same size as the 'green' and 'blue box'
                            fillColor='white', 
                            lineColor='black', 
                            pos=(0, -0.2)
                            )
        self.question_text = visual.TextStim(
                            self.win, 
                            text='?', 
                            color=text_color, 
                            height=0.15, 
                            pos = self.question_box.pos
                            )
         
        # Pre-create objects / stimuli related to the percent display
        self.percent_header = visual.TextStim(
            self.win,
            text= "", # To be updated during trials
            color=text_color, height=0.07, pos=(0, 0.5)
        )
        self.percent_blue_text = visual.TextStim(
            self.win,
            text="", # To be updated during trials
            color='blue', height=0.08, pos=(0.0, 0.38)
        )
        self.percent_green_text = visual.TextStim(
            self.win,
            text="", # To be updated during trials
            color='green', height=0.08, pos=(0.0, 0.28)
        )
        
    # -------------------
    # Instructions
    # -------------------
    def show_instructions(self):
        """Display experiment instructions to the participant"""
        # Create instruction stimuli
        line1 = visual.TextStim(self.win, text="In this task, beads will be drawn from one of two boxes:",
                                color=text_color, height=0.08, pos=(0, 0.45))

        green_box_text = visual.TextStim(self.win, text="Green Box (mostly Green Beads)",
                                         color="green", height=0.08, pos=(-0.47, 0.2))

        blue_box_text = visual.TextStim(self.win, text="Blue Box (mostly Blue Beads)",
                                        color="blue", height=0.08, pos=(0.47, 0.2))

        task_text = visual.TextStim(
            self.win,
            text=("One of these boxes will be chosen RANDOMLY as the Hidden Box.\n"
                  "Beads will be drawn RANDOMLY from it, one at a time WITH REPLACEMENT.\n"
                  "After each bead draw, you will estimate by use of a slider, which box you think more probable to have been chosen as the hidden box.\n"
                  "After 8 beads have been drawn, you will decide which one of the two boxes you think the sample was drawn from."),
            color=text_color, height=0.07, wrapWidth=1.5, pos=(0, -0.2)
        )

        presskey = visual.TextStim(self.win, text="Press any key to continue...",
                                   color="yellow", height=0.08, pos=(0, -0.7))
        
        # Display all instruction elements
        for stim in [line1, green_box_text, blue_box_text, task_text, presskey]:
            stim.draw()

        self.win.flip()
        event.waitKeys()
        
        # Inform the participants of the blocks
        practice_trials_text = visual.TextStim(
            self.win,
            text=("In the main experiment you will complete two blocks of 32 trials each. \n"
                  "In the first block a visual record of beads drawn will be present, to help you keep track of the beads drawn so far. \n"
                  "In the second block, instead of a visual record, you will be informed of the percentwise distribution of blue and green beads drawn so far, which will be updated as beads are drawn. \n"
                  "\n Before beginning the main experiment, you will complete 4 practice trials, to get a grasp on the task. \n"
                  ),
            color=text_color, height=0.07, wrapWidth=1.5, pos=(0, 0)
        )
        
        practice_trials_text.draw()
        presskey.draw()
        self.win.flip()
        event.waitKeys()
        
    # -------------------
    # Trial generator
    # -------------------
    def generate_trials(self):
        trials = []
        
        display_factors = [True, False] # IMPORTANT: CHANGE THIS IN ACCORDANCE WITH THE EXCEL SHEET TRACKING OTHER PARTICIPANT DATA
        
        for display_factor in display_factors:
             # ---- Prepare sequences ----
            # Make copy of all sequences in the 60 ratio
            all_seqs_in_60 = SEQUENCES[60][:]
            mid_60 = len(all_seqs_in_60) // 2

            # Make copy of all sequences in the 90 ratio
            all_seqs_in_90 = SEQUENCES[90][:]
            mid_90 = len(all_seqs_in_90) // 2
            
            # Shuffle all the sequences in 60 ratio (copy), and split the shuffled list in two (first and last)
            random.shuffle(all_seqs_in_60)
            first_seqs_60 = all_seqs_in_60[:mid_60]
            last_seqs_60 = all_seqs_in_60[mid_60:]
            
            # Shuffle all the sequences in 90 ratio (copy), and split the shuffled list in two (first and last)
            random.shuffle(all_seqs_in_90)
            first_seqs_90 = all_seqs_in_90[:mid_90]
            last_seqs_90 = all_seqs_in_90[mid_90:]
            
            # Create ordered sequence chunks matching BLOCK_ORDER
            list_of_sequences = [first_seqs_60, first_seqs_90, last_seqs_60, last_seqs_90] #randomize?
            
            for block_ratio_idx in range(len(BLOCK_ORDER)):
                # Use list comprehension for more efficient sequence processing
                seqs = list_of_sequences[block_ratio_idx][:]
                block_ratio = BLOCK_ORDER[block_ratio_idx]

                # Batch process sequences to avoid repeated calculations
                for seq in seqs:
                    hidden_color = random.choice(['green', 'blue'])
                    # Use pre-cached weights instead of recreating
                    evidence_asymmetry = sum(bead * w for bead, w in zip(seq, self.weights))

                    trials.append({
                        'hidden_color': hidden_color,
                        'display': display_factor,
                        'ratio': block_ratio,
                        'sequence': seq,
                        'prob_estimates': [],
                        'final_choice': None,
                        'evidence_asymmetry': evidence_asymmetry
                    })
        return trials
    
    # -------------------
    # Show boxes + ratios
    # -------------------
    
    # Note: this method doesn't draw, but returns a list of objects to be drawn in the order of first list element to last list element
    def draw_ratio_labels(self, ratio):
        # Update ratio labels
        self.green_ratio_text_top.text = f"{RATIO_MAP[ratio][0]} Green"
        self.green_ratio_text_bottom.text = f"{RATIO_MAP[ratio][1]} Blue"
        self.green_ratio_text_top.pos = (self.left_label.pos[0], self.left_label.pos[1] + 2*0.08)
        self.green_ratio_text_bottom.pos = (self.left_label.pos[0], self.green_ratio_text_top.pos[1] - 0.08)
        
        self.blue_ratio_text_top.text = f"{RATIO_MAP[ratio][0]} Blue"
        self.blue_ratio_text_bottom.text = f"{RATIO_MAP[ratio][1]} Green"
        self.blue_ratio_text_top.pos = (self.right_label.pos[0], self.right_label.pos[1] + 2*0.08)
        self.blue_ratio_text_bottom.pos = (self.right_label.pos[0], self.blue_ratio_text_top.pos[1] - 0.08)
        
        # Put the labels into a list which is returned
        label_list = [self.green_ratio_text_top, 
                      self.green_ratio_text_bottom, 
                      self.blue_ratio_text_top, 
                      self.blue_ratio_text_bottom]
        
        return label_list
    
    # Note: this method doesn't draw, but returns a list of objects to be drawn in the order of first list element to last list element
    def draw_boxes(self, ratio):        
        # Unpack ratio (green-to-blue bead ratio, e.g., [60, 40])
        ratio_green, ratio_blue = RATIO_MAP[ratio]
        
        # Create randomized color assignments for each box
        green_box_colors = ['green'] * ratio_green + ['blue'] * ratio_blue
        blue_box_colors = ['blue'] * ratio_green + ['green'] * ratio_blue
        random.shuffle(green_box_colors)
        random.shuffle(blue_box_colors)
        
        # Start coloring the beads
        for bead, color in zip(self.green_beads, green_box_colors):
            bead.fillColor = color
            bead.lineColor = color
        for bead, color in zip(self.blue_beads, blue_box_colors):
            bead.fillColor = color
            bead.lineColor = color

        # Update frame colors
        self.green_frame.lineColor = 'darkgreen' if ratio_green > ratio_blue else 'darkblue'
        self.blue_frame.lineColor = 'darkblue' if ratio_green > ratio_blue else 'darkgreen'
        
        # Create the list which is going to contain all the recolored objects to be drawn
        # - start with frames to ensure the drawing order is correct (backmost elements drawn first)
        draw_list = [
            self.green_frame, self.blue_frame,
            *self.green_beads, *self.blue_beads,
            self.left_label, self.right_label
        ]
        
        return draw_list

    # -------------------
    # Draw visual bead display
    # -------------------
    def draw_display(self, drawn_beads, hidden_color):
        """Optimized bead display using pre-calculated values and cached circles"""
        # Use pre-cached values instead of recalculating
        start_x = -(self.total_beads - 1) * self.bead_spacing / 2
        y_pos = 0.4

        # Pre-calculate colors
        majority_color = hidden_color
        minority_color = 'green' if majority_color == 'blue' else 'blue'

        for i in range(self.total_beads):
            pos_x = start_x + i * self.bead_spacing
            if i < len(drawn_beads):
                bead_color = majority_color if drawn_beads[i] == 1 else minority_color
                # Use cached circle and update position/color
                circle = self.bead_circles[bead_color]
                circle.pos = (pos_x, y_pos)
            else:
                # Use cached empty circle
                circle = self.bead_circles['empty']
                circle.pos = (pos_x, y_pos)
            circle.draw()

    # -------------------
    # Draw numeric display (percent of total 8)
    # -------------------
    def draw_numeric_display(self, drawn_beads, hidden_color):
        """Optimized numeric display with pre-calculated values"""
        # Use pre-cached total_beads value
        majority_color = hidden_color
        minority_color = 'green' if majority_color == 'blue' else 'blue'

        # More efficient counting using sum() with generator expression
        num_majority_drawn = sum(1 for b in drawn_beads if b == 1)
        num_minority_drawn = len(drawn_beads) - num_majority_drawn  # Avoid second sum

        # Pre-calculate percentages
        majority_percent = (num_majority_drawn / len(drawn_beads)) * 100
        minority_percent = (num_minority_drawn / len(drawn_beads)) * 100

        # Optimize color assignment
        if majority_color == 'blue':
            blue_percent, green_percent = majority_percent, minority_percent
        else:
            blue_percent, green_percent = minority_percent, majority_percent
        
        # Update the text of percent display according to trial information
        self.percent_header.text = f"Percentwise distribution of the {len(drawn_beads)}/8 beads drawn so far:"
        self.percent_blue_text.text = f"Blue: {blue_percent:.1f}%"
        self.percent_green_text.text =  f"Green: {green_percent:.1f}%"
        
        for text in [self.percent_header, self.percent_blue_text, self.percent_green_text]:
            text.draw()

    # -------------------
    # Run one trial
    # -------------------
    def run_trial(self, trial_num, practice):
        # Determine whether the trial being run is practice or not (we're not saving practice data)
        if practice == True:
            trial = self.practice_trials[trial_num]
        else:
            trial = self.trials[trial_num]

        # Show boxes and their appropriate labels (in Ashinoff Fig 1a: 'Trial Start')
        label_list = self.draw_ratio_labels(trial['ratio'])
        
        # Forgive the name: 'list_of_box_objects' contains a list of all the drawable objects making up the left and right box
        list_of_box_objects = self.draw_boxes(trial['ratio'])
        
        # Move the boxes close together when they are first seen
        displacement = 0.20
        for object in (label_list + list_of_box_objects):
            if object.pos[0] < 0:
                object.pos = (object.pos[0] + displacement, object.pos[1])
            elif object.pos[0] > 0:
                object.pos = (object.pos[0] - displacement, object.pos[1])
            object.draw()
        
        self.win.flip()
        core.wait(3.0)
        
        # Transition to prior rating screen
        self.win.flip()
        core.wait(0.5)  
        
        # Move the two boxes further apart again (such that the slider fits inbetween them)
        for object in (label_list + list_of_box_objects):
            if object.pos[0] < 0:
                object.pos = (object.pos[0] - displacement, object.pos[1])
            elif object.pos[0] > 0:
                object.pos = (object.pos[0] + displacement, object.pos[1])
        
        # This sort of takes a 'screenshot' of the two boxes to be drawn, and those screenshots are drawn, rather than 200+ elements
        static_stim = visual.BufferImageStim(self.win, stim=list_of_box_objects + label_list)
        
        # Prior rating - optimized with cached stimuli (in Ashinoff Fig 1a: 'Draw (0)')
        self.mouse.clickReset()
        self.slider.reset()
        prior_collected = False
        while not prior_collected:
            # Cursor control of the slider
            mouse_x = self.mouse.getPos()[0]
            slider_min = self.slider.pos[0] - self.slider.size[0] / 2
            slider_max = self.slider.pos[0] + self.slider.size[0] / 2
                
            # Map mouse x to slider range [0,1] (clipped)
            hover_value = np.clip((mouse_x - slider_min) / (slider_max - slider_min), 0, 1)
            hover_x = slider_min + hover_value * (slider_max - slider_min)
                        
            # Update marker bar position
            self.marker_bar.pos = (hover_x, -0.4)
                
            # Compute left/right probabilities and draw the slider end labels dynamically
            right_prob = int(round(hover_value * 100))
            left_prob = 100 - right_prob
            
            self.slider_left_label.text = f'{left_prob}%'
            self.slider_left_label.pos = (slider_min, -0.55)
            self.slider_right_label.text = f'{right_prob}%'
            self.slider_right_label.pos = (slider_max, -0.55)
            
            # Draw the two boxes again
            static_stim.draw()
            
            # Draw prior text, slider and slider related elements (marker bar and labels) 
            self.prior_text.draw()
            self.slider.draw()
            self.marker_bar.draw()
            self.slider_left_label.draw()
            self.slider_right_label.draw()
                
            self.win.flip()

            rating = self.slider.getRating()
            if rating is not None:
                trial['prob_estimates'].append(rating)
                prior_collected = True

            # Optimized escape key handling
            if 'escape' in event.getKeys(keyList=['escape']):
                self.win.close()
                self.save_results() 
                core.quit()

        # Bead sequence - optimized with cached stimuli
        majority_color = trial['hidden_color']
        minority_color = 'green' if majority_color == 'blue' else 'blue'
        
        # (The following Corresponds to Ashinoff Fig 1a: 'Draw (1) + Estimate (1) + ... + Draw (8) + Estimate (8))
        for idx, bead in enumerate(trial['sequence']):
            # Draw the white question mark box, before having the bead rise from it
            self.question_box.draw()
            self.question_text.draw()
            self.win.flip()
            
            # Determine bead color and animate bead rising from the box
            bead_color = majority_color if bead == 1 else minority_color
            bead_stim = self.bead_circles[bead_color]
            
            # Animation parameters and clock
            clock = core.Clock()
            duration = 0.8      # total time of movement in seconds
            start_y = self.question_box.pos[1] + (self.question_box.height / 2) - (bead_stim.radius * 2)      # starts right below top edge of question box (barely visible)
            end_y = start_y + 0.4         # rises to top position some space above the box

            # Time-based animation loop
            while True:
                t = clock.getTime() / duration  # normalized time 0->1
                if t >= 1.0:
                    break  # exit loop after reaching the end
                
                # sine-in-out easing
                y_pos = start_y + (end_y - start_y) * 0.5 * (1 - math.cos(math.pi * t))
                bead_stim.pos = (0, y_pos)
                
                # Draw in order: bead, box (so bead is behind the box in the beginning)
                bead_stim.draw()
                self.question_box.draw()
                self.question_text.draw()
                self.win.flip()

            # Hold bead at top of the question box briefly
            self.question_box.draw()
            self.question_text.draw()
            bead_stim.pos = (0, end_y)
            bead_stim.draw()
            self.win.flip()
            core.wait(0.3)

            # Rating after bead - optimized with cached stimuli
            self.mouse.clickReset()
            self.slider.reset()
            rating_collected = False
            while not rating_collected:
                # Cursor control of the slider
                mouse_x = self.mouse.getPos()[0]
                slider_min = self.slider.pos[0] - self.slider.size[0] / 2
                slider_max = self.slider.pos[0] + self.slider.size[0] / 2
                
                # Map mouse x to slider range [0,1] (clipped)
                hover_value = np.clip((mouse_x - slider_min) / (slider_max - slider_min), 0, 1)
                hover_x = slider_min + hover_value * (slider_max - slider_min)
                
                # Update marker bar position
                self.marker_bar.pos = (hover_x, -0.4)
                    
                # Compute left/right probabilities and draw the slider end labels dynamically
                right_prob = int(round(hover_value * 100))
                left_prob = 100 - right_prob
                
                self.slider_left_label.text = f'{left_prob}%'
                self.slider_left_label.pos = (slider_min, -0.55)
                self.slider_right_label.text = f'{right_prob}%'
                self.slider_right_label.pos = (slider_max, -0.55)
                
                # Draw the two boxes again
                static_stim.draw()
                
                # Draw prior text, slider and slider related elements (marker bar and labels) 
                self.est_prob_text.draw()
                self.slider.draw()
                self.marker_bar.draw()
                self.slider_left_label.draw()
                self.slider_right_label.draw()
                
                # Select the display of beads (visual or percentage format) according to the block
                if trial['display']:
                    self.draw_display(trial['sequence'][:idx+1], majority_color)
                else:
                    self.draw_numeric_display(trial['sequence'][:idx+1], majority_color)
                
                self.win.flip()
                
                # Collect rating
                rating = self.slider.getRating()
                if rating is not None:
                    trial['prob_estimates'].append(rating)
                    rating_collected = True

                # Optimized escape key handling
                if 'escape' in event.getKeys(keyList=['escape']):
                    self.win.close()
                    self.save_results() 
                    core.quit()
        
        # Final choice 
        
        # Draw the boxes
        static_stim.draw()
        
        # Draw the bead record / final percent distribution 
        if trial['display']:
            self.draw_display(trial['sequence'][:8], majority_color)
        else:
            self.draw_numeric_display(trial['sequence'][:8], majority_color)

        # Draw the boxes again with final choice text
        self.final_choice_text1.draw()
        
        # Show it
        self.win.flip()
        
        # Record the answer
        keys = event.waitKeys(keyList=['left', 'right', 'escape'])
        if 'escape' in keys:
            self.win.close()
            self.save_results()
            core.quit()
        elif keys[0] == 'left':
            trial['final_choice'] = 'green'
            # Change the border color of the chosen box to yellow and draw the boxes again. 
            for object in list_of_box_objects:
                if (isinstance(object, visual.Rect) and getattr(object, 'color_id', '') == 'green'):
                    object.lineColor = 'yellow'
                object.draw()
            
            # Draw the display again
            if trial['display']:
                self.draw_display(trial['sequence'][:8], majority_color)
            else:
                self.draw_numeric_display(trial['sequence'][:8], majority_color)
            
            # Draw the final choice text
            self.final_choice_text1.draw()
            
            self.win.flip()
            core.wait(0.3)
        elif keys[0] == 'right':
            trial['final_choice'] = 'blue'
            
             # Change the border color of the chosen box to yellow and draw the boxes again. 
            for object in list_of_box_objects:
                if (isinstance(object, visual.Rect) and getattr(object, 'color_id', '') == 'blue'):
                    object.lineColor = 'yellow'
                object.draw()
                    
            # Draw the display again
            if trial['display']:
                self.draw_display(trial['sequence'][:8], majority_color)
            else:
                self.draw_numeric_display(trial['sequence'][:8], majority_color)
            
            # Draw the final choice text
            self.final_choice_text1.draw()
            self.win.flip()
            core.wait(0.3)
    
        if (practice == False): 
            self.results.append(trial)
        
        self.win.flip()
        core.wait(0.5)
    
    # -------------------
    # Show break text
    # -------------------
    def show_break(self, duration=60, message="Pause"):
        break_text = visual.TextStim(
            self.win, 
            text=message, 
            color=text_color, height=0.07, wrapWidth=1.5
        )
        
        break_text.draw()
        self.win.flip()
        core.wait(duration)
        
        break_over_text = visual.TextStim(
            self.win, 
            text="Get ready. The next block will begin in a few seconds.", 
            color=text_color, height=0.07, wrapWidth=1.5
        )
        
        break_over_text.draw()
        self.win.flip()
        core.wait(5.0)
        
    # -------------------
    # Generate practice trials
    # -------------------
    
    def generate_practice_trials(self):
        prac_trials = []
        
        for display_factor in [True, False]:
            for block_ratio in [60,90]:
                
                # Pick a random given sequence from the list of sequences belonging to the currently chosen block ratio
                random_seq = random.choice(SEQUENCES[block_ratio])
                hidden_color = random.choice(['green', 'blue'])
                evidence_asymmetry = sum(bead * w for bead, w in zip(random_seq, self.weights))
                
                prac_trials.append({
                        'hidden_color': hidden_color,
                        'display': display_factor,
                        'ratio': block_ratio,
                        'sequence': random_seq,
                        'prob_estimates': [],
                        'final_choice': None,
                        'evidence_asymmetry': evidence_asymmetry
                    })
        return prac_trials
                
    # -------------------
    # Run main experiment
    # -------------------
    def run_experiment(self):
         # First, run the 4 practice trials
        practice_trial_text = visual.TextStim(
            self.win, 
            text="Practice trials",
            color=text_color, height=0.08
        )
        practice_trial_text.draw()
        win.flip()
        core.wait(1.5) 
       
        for trial_num in range(len(self.practice_trials)): 
            self.run_trial(trial_num, practice = True)
        
        # Transition from the practice block to the main experiment
        win.flip()
        core.wait(2.0)
        transition_text = visual.TextStim(
            self.win, 
            text="You've completed the practice trials.\n Please consult the experimenter if you have any questions. \n If not, press any key to begin the main experiment.",
            color=text_color, height=0.08
        )
        transition_text.draw()
        self.win.flip()
        event.waitKeys()
        
        # Begin the main experiment
        block_structure = BLOCK_ORDER # [60, 90, 60, 90]
        trials_per_block = 8
        num_blocks_per_display = len(block_structure) # 4
        
        block_counter = 0 # IMPORTANT: Should be 0 for the actual experiment
        trial_counter = 0 # IMPORTANT: Should be 0 for the actual experiment
        display_switch_done = False
        
        # Iterate over all main trials
        n_trials = len(self.trials)
        for trial_num in range(0,n_trials): # IMPORTANT: should be range (n_trials) for the actual experiment
            self.run_trial(trial_num, practice = False)
            trial_counter += 1
            
            # Check if we've finished the current block
            if trial_counter == trials_per_block:
                trial_counter = 0 # Reset for next block
                block_counter += 1
            
                # Determine type of break
                if block_counter == num_blocks_per_display:
                    # We're transitioning from 'display = True' to 'display = False'
                    if not display_switch_done:
                        self.show_break(duration=60, message = "You've completed the first block. \n The next (and final) block will begin in 1 minute.\n In each of the following trials, the display format of beads drawn will be different. \n A notification will appear on screen when the break is over.")
                        display_switch_done = True
                elif block_counter < num_blocks_per_display * 2: # If the block is in between 1 and 8, but not the 5th block (transition)
                    # Regular 1-minute break between ratio blocks
                    self.show_break(duration = 60, message="1-minute break.\n A notification will appear on screen when the break is over.")
                    
    # -------------------
    # Save results
    # -------------------
    def save_results(self, filename=None):
        if filename is None:
            filename = f"beads_task_results_{self.subject_id}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(['Subject', 'Trial', 'HiddenColor', 'Display', 'Ratio',
                             'BeadPosition', 'Sequence', 'MajorityBeads', 'ProbEstimate', 
                             'FinalChoice', 'EvidenceAsymmetry', 'Accuracy'])
            
            for t_num, t in enumerate(self.results, start=1):
                # Convert sequence to string for better readability
                sequence_str = ','.join(map(str, t['sequence']))
                
                # Calculate number of majority beads (1s in the sequence)
                num_majority_beads = sum(t['sequence'])
                accuracy = int(t['hidden_color'] == t['final_choice'])
                
                # Create a row for the prior estimate (bead position 0)
                writer.writerow([
                    self.subject_id, t_num, t['hidden_color'], t['display'], t['ratio'],
                    0, sequence_str, num_majority_beads, t['prob_estimates'][0], 
                    t['final_choice'], t['evidence_asymmetry'], accuracy
                ])
                
                # Create rows for each bead and subsequent probability estimate
                for bead_pos, prob_estimate in enumerate(t['prob_estimates'][1:], 1):
                    writer.writerow([
                        self.subject_id, t_num, t['hidden_color'], t['display'], t['ratio'],
                        bead_pos, sequence_str, num_majority_beads, prob_estimate,
                        t['final_choice'], t['evidence_asymmetry'], accuracy
                    ])

# -------------------
# Run everything
# -------------------

exp = BeadsTask(win, subject_number)    
exp.show_instructions()  
exp.run_experiment()
exp.save_results()


thanks = visual.TextStim(win, text="Thank you :)", color="black", height=0.1)
thanks.draw()
win.flip()
core.wait(2)

win.close()
core.quit()