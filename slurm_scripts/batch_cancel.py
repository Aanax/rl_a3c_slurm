import argparse
import os

def isInt(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cancel Slurm jobs listed in a file. Reads job IDs from the specified file and cancels each valid job ID using scancel.'
    )
    parser.add_argument('job_file', type=str, help='Path to the file containing job IDs (one per line)')
    args = parser.parse_args()

    with open(args.job_file, 'r') as f:
        for line in f:
            for word in line.split():
                if isInt(word):
                    os.system('scancel %d' % int(word))
