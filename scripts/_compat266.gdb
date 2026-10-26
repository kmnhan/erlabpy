set pagination off
set confirm off
set print pretty on
set print elements 30
set print max-depth 5
handle SIGPIPE nostop noprint pass

python
import gdb

def capture_native_failure(event):
    if not isinstance(event, gdb.SignalEvent):
        return
    print("\nCOMPAT266 NATIVE FAILURE: " + event.stop_signal)
    gdb.execute("bt full 80")
    try:
        gdb.execute("py-bt")
    except gdb.error as exc:
        print("Python backtrace unavailable: " + str(exc))
    gdb.execute("thread apply all bt 30")
    gdb.execute("info sharedlibrary")

gdb.events.stop.connect(capture_native_failure)
end

run
