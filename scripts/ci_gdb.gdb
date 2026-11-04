# Run once and preserve the test process exit code.
set print thread-events off
run
if !$_isvoid($_exitcode)
    quit $_exitcode
end

# A native stop has no exit code. Save every thread stack and fail the job.
thread apply all -c backtrace full
info proc status
quit 1
