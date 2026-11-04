# Run once and preserve the test process exit code.
set pagination off
set confirm off
set print thread-events off
run
if !$_isvoid($_exitcode)
    quit $_exitcode
end

# A native stop has no exit code. Save every thread stack and fail the job.
thread apply all -c backtrace full
info sharedlibrary
info proc mappings
info proc status
info registers
x/16i $pc-32

# Let Python's fatal signal handler print its thread stacks before termination.
# GDB stops before faulthandler runs, so quitting here loses the Python frames.
handle SIGSEGV nostop print pass
handle SIGABRT nostop print pass
continue
quit 1
