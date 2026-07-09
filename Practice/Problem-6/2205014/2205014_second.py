import importlib

first_part=importlib.import_module("2205014_first")

Signal=first_part.Signal
LTI_System=first_part.LTI_System

#read file
f=open("input_signal.txt","r")
lines=f.readlines()
n_start,n_end=map(int,lines[0].split())
INF=max(abs(n_start),abs(n_end))
noisy_signal=Signal(INF)
noisy_signal.x=list(map(int,lines[1].split()))
# print(noisy_signal.x)
noisy_signal.plot("Noisy Signal")

#smoothing signal

impulse_response=Signal(2)
impulse_response.set_value_at_time(-2,1/5.0)
impulse_response.set_value_at_time(-1,1/5.0)
impulse_response.set_value_at_time(0,1/5.0)
impulse_response.set_value_at_time(1,1/5.0)
impulse_response.set_value_at_time(2,1/5.0)
System=LTI_System(impulse_response)

y=System.output(noisy_signal)
y.plot("Smoothed Signal")