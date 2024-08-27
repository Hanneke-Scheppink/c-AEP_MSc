import os
import numpy as np
import pandas as pd

# track_orders = np.array([[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]])  # balanced latin square
track_orders = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])  # order of tracks
tracks = np.array([[1, 5], [2, 6], [3, 7], [4, 8]])  # tracks (stories), note paired on left/right
track_parts = np.array([1, 2, 3, 4])  # parts of tracks (stories)
design_dir = r"D:\Users\bci\caep\experiment\parallel\designs"

i = 0  # count the number of generated designs
designs = []  # keep track of designs

# Presented attended story on both sides
for i_attention in range(2):

    # Present stories starting on both sides
    for i_track_side in range(2):

        # Present stories startuing with modulated and unmodulated
        for i_start_condition in range(2):

            # Present codes on both sides
            for i_code_side in range(2):
                if i_code_side == 0:
                    code_left = 0
                    code_right = 1
                else:
                    code_left = 1
                    code_right = 0

                # Present tracks in counter-balanced order
                for i_order in range(len(track_orders)):

                    left = []
                    right = []
                    attention = []

                    # Loop all tracks
                    for i_track in range(len(tracks)):

                        # Put tracks on left and right
                        if i_track_side == 0:
                            track_left = tracks[track_orders[i_order][i_track]][0]
                            track_right = tracks[track_orders[i_order][i_track]][1]
                        else:
                            track_left = tracks[track_orders[i_order][i_track]][1]
                            track_right = tracks[track_orders[i_order][i_track]][0]

                        # Loop all parts
                        for i_part in range(len(track_parts)):

                            # Always in order
                            part = track_parts[i_part]
                            
                            # Part 1 and 2 presented on the one side, part 3 and 4 on the other side
                            if part == 3:  # do only once, affects part 4 too
                                track_left, track_right = track_right, track_left

                            # Add audio with alternating modulated and unmodulated
                            if i_start_condition == 0 and i_part % 2 == 0 or i_start_condition == 1 and i_part % 2 == 1:
                                left.append(f"t{track_left:d}_p{part:d}_c{code_left:d}")
                                right.append(f"t{track_right:d}_p{part:d}_c{code_right:d}")
                            else:
                                left.append(f"t{track_left:d}_p{part:d}")
                                right.append(f"t{track_right:d}_p{part:d}")

                            # Set attention left or right, same for two subsequent parts
                            if i_attention == 0 and part <= 2 or i_attention == 1 and part >= 3:
                                attention.append("left")
                            else:
                                attention.append("right")

                    # Keep track of designs
                    i += 1
                    designs.append("-".join(left) + "|" + "-".join(right) + "|" + "-".join(attention))

                    # Save in accessible format
                    df = pd.DataFrame({"left": left, "right": right, "attention": attention})
                    df.index += 1  # start counting at 1 for trial number
                    df.to_csv(os.path.join(design_dir, f"sub-{i:02d}.csv"))
                    print("\n" + 20 * "*" + f" sub-{i:02d} " + 20 * "*")
                    print(df)

print("\n" + 50 * "*")
print("Number of generated designs:", i)
print("Number of unique designs:", len(set(designs)))
