import multiprocessing as mp
import sys


def main():
    mp.set_start_method("spawn", force=True)
    import slam

    slam.cli_main(sys.argv[1:])


if __name__ == "__main__":
    main()
