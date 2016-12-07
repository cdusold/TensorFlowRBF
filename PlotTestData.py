from matplotlib import pyplot as plt
import numpy as np

results = np.load("feedforwardtimings.npy")

#Raw Timings Plot Feedforward
plt.figure()
plt.suptitle("Feedforward", fontsize=24, y=1.05)
plt.subplot(2,2,1)
plt.title("Timing With Ten by Ten Sized Matrices")
plt.plot(results[:-1,0])
plt.scatter([5],results[-1:,0])
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.subplot(2,2,2)
plt.title("Timing With a Hundred by Hundred Sized Matrices")
plt.plot(results[:-1,1])
plt.scatter([5],results[-1:,1])
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.subplot(2,2,3)
plt.title("Timing With a Thousand by a Thousand Sized Matrices")
plt.plot(results[:-1,2])
plt.scatter([5],results[-1:,2])
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.subplot(2,2,4)
plt.title("All Timings In Log Scale")
plt.plot(results[:-1])
plt.scatter([5],results[-1:,0],color="blue")
plt.scatter([5],results[-1:,1], color="green")
plt.scatter([5],results[-1:,2], color="red")
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.yscale("log")
plt.tight_layout()

plt.savefig("feedforward.pgf")
plt.show()

#Relative Timings Feedforward
thread_counts = np.array([1,2,4,8,16,32*256])
results = results[0,None]/results
plt.figure()
plt.suptitle("Feedforward", fontsize=24, y=1.05)
plt.subplot(2,2,1)
plt.title("All Speed Ups")
plt.plot(results[:-1])
plt.scatter([5],results[-1:,0],color="blue")
plt.scatter([5],results[-1:,1], color="green")
plt.scatter([5],results[-1:,2], color="red")
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Speed up ratio on one calculation (log scale)")
plt.yscale("log")
plt.legend(["10x10","100x100","1000x1000"],loc=2)

plt.subplot(2,2,2)
plt.title("CPU Only Speed Ups")
plt.plot(results[:-1]-1)
plt.xticks(range(-1,6),["","1", "2", "4", "8", "16", ""])
plt.xlabel("Number of threads")
plt.ylabel("Relative speed difference on one calculation")
plt.legend(["10x10","100x100","1000x1000"],loc=2)

plt.subplot(2,2,3)
plt.title("Speed Up Per Thread")
plt.plot((results[1:-1]-1)/thread_counts[1:-1,None])
plt.scatter([4],(results[-1:,0]-1)/thread_counts[-1],color="blue")
plt.scatter([4],(results[-1:,1]-1)/thread_counts[-1], color="green")
plt.scatter([4],(results[-1:,2]-1)/thread_counts[-1], color="red")
plt.xticks(range(-1,6),["", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Relative speed difference on one calculation")
plt.legend(["10x10","100x100","1000x1000"],loc=1)

plt.subplot(2,2,4)
def amdahlPortion(speedup,threads):
    return threads*(speedup-1)/((threads-1)*speedup)
plt.title("Amdahl's Law Calculated Parallelizable Portion")
plt.plot(amdahlPortion(results[1:-1],thread_counts[1:-1,None]))
plt.scatter([4],amdahlPortion(results[-1:,0],thread_counts[-1]),color="blue")
plt.scatter([4],amdahlPortion(results[-1:,1],thread_counts[-1]), color="green")
plt.scatter([4],amdahlPortion(results[-1:,2],thread_counts[-1]), color="red")
plt.xticks(range(-1,6),["", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Ratio of parallelizable code to total code")
plt.legend(["10x10","100x100","1000x1000"],loc=10)

plt.tight_layout()
plt.savefig("feedforward2.pgf")
plt.show()

#Backprop time
results = np.load("backproptimings.npy")

#Raw Timings Plot Backpropagation
plt.figure()
plt.suptitle("Backpropagation", fontsize=24, y=1.05)
plt.subplot(2,2,1)
plt.title("Timing With Ten by Ten Sized Matrices")
plt.plot(results[:-1,0])
plt.scatter([5],results[-1:,0])
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.subplot(2,2,2)
plt.title("Timing With a Hundred by Hundred Sized Matrices")
plt.plot(results[:-1,1])
plt.scatter([5],results[-1:,1])
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.subplot(2,2,3)
plt.title("Timing With a Thousand by a Thousand Sized Matrices")
plt.plot(results[:-1,2])
plt.scatter([5],results[-1:,2])
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")

plt.subplot(2,2,4)
plt.title("All Timings In Log Scale")
plt.plot(results[:-1])
plt.scatter([5],results[-1:,0],color="blue")
plt.scatter([5],results[-1:,1], color="green")
plt.scatter([5],results[-1:,2], color="red")
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Seconds to compute one calculation")
plt.yscale("log")

plt.tight_layout()

plt.savefig("backprop.pgf")
plt.show()

#Relative Timings Backpropagation
results = results[0,None]/results
plt.figure()
plt.suptitle("Feedforward", fontsize=24, y=1.05)
plt.subplot(2,2,1)
plt.title("All Speed Ups")
plt.plot(results[:-1])
plt.scatter([5],results[-1:,0],color="blue")
plt.scatter([5],results[-1:,1], color="green")
plt.scatter([5],results[-1:,2], color="red")
plt.xticks(range(-1,7),["","1", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Speed up ratio on one calculation (log scale)")
plt.yscale("log")
plt.legend(["10x10","100x100","1000x1000"],loc=2)

plt.subplot(2,2,2)
plt.title("CPU Only Speed Ups")
plt.plot(results[:-1]-1)
plt.xticks(range(-1,6),["","1", "2", "4", "8", "16", ""])
plt.xlabel("Number of threads")
plt.ylabel("Relative speed difference on one calculation")
plt.legend(["10x10","100x100","1000x1000"],loc=2)

plt.subplot(2,2,3)
plt.title("Speed Up Per Thread")
plt.plot((results[1:-1]-1)/thread_counts[1:-1,None])
plt.scatter([4],(results[-1:,0]-1)/thread_counts[-1],color="blue")
plt.scatter([4],(results[-1:,1]-1)/thread_counts[-1], color="green")
plt.scatter([4],(results[-1:,2]-1)/thread_counts[-1], color="red")
plt.xticks(range(-1,6),["", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Relative speed difference on one calculation")
plt.legend(["10x10","100x100","1000x1000"],loc=1)

plt.subplot(2,2,4)
plt.title("Amdahl's Law Calculated Parallelizable Portion")
plt.plot(amdahlPortion(results[1:-1],thread_counts[1:-1,None]))
plt.scatter([4],amdahlPortion(results[-1:,0],thread_counts[-1]),color="blue")
plt.scatter([4],amdahlPortion(results[-1:,1],thread_counts[-1]), color="green")
plt.scatter([4],amdahlPortion(results[-1:,2],thread_counts[-1]), color="red")
plt.xticks(range(-1,6),["", "2", "4", "8", "16", "GPU", ""])
plt.xlabel("Number of threads (or GPU)")
plt.ylabel("Ratio of parallelizable code to total code")
plt.legend(["10x10","100x100","1000x1000"],loc=10)

plt.tight_layout()
plt.savefig("feedforward2.pgf")
plt.show()