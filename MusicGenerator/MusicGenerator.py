import random
import os
import time
from datetime import datetime
from typing import Dict, List

from midiutil import MIDIFile
from pyo import *

from Algorithm import generate_genome, Genome, selection_pair, single_point_crossover, mutation

# Constants
BITS_PER_NOTE = 4
KEYS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
SCALES = ["major", "minorM", "dorian", "phrygian", "lydian", "mixolydian", "majorBlues", "minorBlues"]

# Helper function to convert a list of bits to an integer
def int_from_bits(bits: List[int]) -> int:
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))

# Function to prompt for user input with optional default value and type casting
def prompt_for_input(prompt: str, default=None, type_=None):
    user_input = input(prompt + (f" [{default}]" if default is not None else "") + ": ")
    if user_input == "" and default is not None:
        return default
    if type_ is not None:
        try:
            return type_(user_input)
        except ValueError:
            print("Invalid input. Please try again.")
            return prompt_for_input(prompt, default, type_)
    return user_input

# Function to convert a genome to melody representation
def genome_to_melody(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: int, key: str, scale: str, root: int) -> Dict[str, list]:
    # Extract notes from the genome
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]

    note_length = 4 / float(num_notes)

    # Define the scale
    scl = EventScale(root=key, scale=scale, first=root)

    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }

    # Convert notes to melody representation
    for note in notes:
        integer = int_from_bits(note)

        if pauses:
            integer = int(integer % pow(2, BITS_PER_NOTE - 1))

        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody["notes"] += [0]
            melody["velocity"] += [0]
            melody["beat"] += [note_length]
        else:
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"] += [integer]
                melody["velocity"] += [127]
                melody["beat"] += [note_length]

    steps = []
    for step in range(num_steps):
        steps.append([scl[(note+step*2) % len(scl)] for note in melody["notes"]])

    melody["notes"] = steps
    return melody

# Function to convert a genome to events for sound synthesis
def genome_to_events(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: bool, key: str, scale: str, root: int, bpm: int) -> [Events]:
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    return [
        Events(
            midinote=EventSeq(step, occurrences=1),
            midivel=EventSeq(melody["velocity"], occurrences=1),
            beat=EventSeq(melody["beat"], occurrences=1),
            attack=0.001,
            decay=0.05,
            sustain=0.5,
            release=0.005,
            bpm=bpm
        ) for step in melody["notes"]
    ]

# Function to evaluate the fitness of a genome
def fitness(genome: Genome, s: Server, num_bars: int, num_notes: int, num_steps: int,
            pauses: bool, key: str, scale: str, root: int, bpm: int) -> int:
    m = metronome(bpm)

    events = genome_to_events(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
    for e in events:
        e.play()
    s.start()

    rating = input("Rating (0-5)")

    for e in events:
        e.stop()
    s.stop()
    time.sleep(1)

    try:
        rating = int(rating)
    except ValueError:
        rating = 0

    return rating

# Function to generate a metronome sound
def metronome(bpm: int):
    met = Metro(time=1 / (bpm / 60.0)).play()
    t = CosTable([(0, 0), (50, 1), (200, .3), (500, 0)])
    amp = TrigEnv(met, table=t, dur=.25, mul=1)
    freq = Iter(met, choice=[660, 440, 440, 440])
    return Sine(freq=freq, mul=amp).mix(2).out()

# Function to save the melody generated from a genome to MIDI file
def save_genome_to_midi(filename: str, genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                        pauses: bool, key: str, scale: str, root: int, bpm: int):
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale, root)

    if len(melody["notes"][0]) != len(melody["beat"]) or len(melody["notes"][0]) != len(melody["velocity"]):
        raise ValueError

    mf = MIDIFile(1)

    track = 0
    channel = 0

    time = 0.0
    mf.addTrackName(track, time, "Sample Track")
    mf.addTempo(track, time, bpm)

    for i, vel in enumerate(melody["velocity"]):
        if vel > 0:
            for step in melody["notes"]:
                mf.addNote(track, channel, step[i], time, melody["beat"][i], vel)

        time += melody["beat"][i]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        mf.writeFile(f)

# Main function
def main():
    # Prompt user for various parameters
    num_bars = prompt_for_input("Number of bars", default=8, type_=int)
    num_notes = prompt_for_input("Notes per bar", default=4, type_=int)
    num_steps = prompt_for_input("Number of steps", default=1, type_=int)
    pauses = prompt_for_input("Introduce Pauses (True/False)", default=False, type_=bool)
    key = prompt_for_input("Key", default="C")
    scale = prompt_for_input("Scale", default="major")
    root = prompt_for_input("Scale Root", default=4, type_=int)
    population_size = prompt_for_input("Population size", default=10, type_=int)
    num_mutations = prompt_for_input("Number of mutations", default=2, type_=int)
    mutation_probability = prompt_for_input("Mutation probability", default=0.5, type_=float)
    bpm = prompt_for_input("BPM", default=128, type_=int)

    folder = str(int(datetime.now().timestamp()))

    # Generate initial population
    population = [generate_genome(num_bars * num_notes * BITS_PER_NOTE) for _ in range(population_size)]

    s = Server().boot()

    population_id = 0

    running = True
    while running:
        # Shuffle the population
        random.shuffle(population)

        # Evaluate fitness for each genome in the population
        population_fitness = [(genome, fitness(genome, s, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)) for genome in population]

        # Sort the population based on fitness
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)

        # Select top genomes for next generation
        next_generation = population[0:2]

        for j in range(int(len(population) / 2) - 1):

            # Function to look up fitness of a genome
            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0

            # Selection, crossover, and mutation
            parents = selection_pair(population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, num=num_mutations, probability=mutation_probability)
            offspring_b = mutation(offspring_b, num=num_mutations, probability=mutation_probability)
            next_generation += [offspring_a, offspring_b]

        print(f"population {population_id} done")

        # Playback top two genomes
        events = genome_to_events(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("here is the no1 hit …")
        s.stop()
        for e in events:
            e.stop()

        time.sleep(1)

        events = genome_to_events(population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        for e in events:
            e.play()
        s.start()
        input("here is the second best …")
        s.stop()
        for e in events:
            e.stop()

        time.sleep(1)

        # Save MIDI files for each genome in the population
        print("saving population midi …")
        for i, genome in enumerate(population):
            save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        print("done")

        # Prompt user to continue or stop
        running = input("continue? [Y/n]") != "n"
        population = next_generation
        population_id += 1


if __name__ == '__main__':
    main()
