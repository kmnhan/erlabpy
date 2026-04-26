import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from erlab.interactive.imagetool.manager import main

    main()
