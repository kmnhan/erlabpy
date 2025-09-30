import multiprocessing

from erlab.interactive.imagetool.manager import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
