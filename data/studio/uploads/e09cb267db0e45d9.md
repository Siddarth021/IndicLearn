# Signals and Systems — Study Notes

These notes cover the core ideas of an introductory Signals and Systems course for second-year
electronics and electrical engineering students.

## Signals

### Continuous-Time and Discrete-Time Signals
A **signal** is a function that carries information about a physical quantity such as voltage, sound pressure or temperature. A **continuous-time signal** is defined for every value of time t, and it is written x(t). A **discrete-time signal** is defined only at integer instants n, and it is written x[n]. For example, a speech waveform recorded by a microphone is continuous-time, while the list of samples stored in a computer is discrete-time.

### Basic Signals
The **unit impulse** is a signal that is zero everywhere except at the origin, where it is concentrated with unit area. The **unit step** is a signal that is zero for negative time and one for positive time. The unit step is the running integral of the unit impulse. A **sinusoid** is a signal of the form A cos(wt + phi), where A is the amplitude, w is the angular frequency and phi is the phase.

x(t) = A cos(w t + phi)

### Energy and Power
The **energy** of a signal is the integral of the squared magnitude of the signal over all time. A signal with finite energy is called an **energy signal**. A **power signal** is a signal with finite average power and infinite energy. For example, a periodic sinusoid is a power signal because its energy over infinite time is unbounded.

## Systems

### System Properties
A **system** is a process that transforms an input signal into an output signal. A system is **linear** if it satisfies superposition, which means that the response to a weighted sum of inputs is the same weighted sum of the responses. A system is **time-invariant** if a delay in the input causes the same delay in the output. **Causality** means that the output at any time depends only on present and past inputs. A system is **BIBO stable** if every bounded input produces a bounded output.

### LTI Systems
A **linear time-invariant system**, abbreviated LTI system, is a system that is both linear and time-invariant. The behaviour of an LTI system is completely described by its **impulse response**, which is the output of the system when the input is the unit impulse. Because of superposition, any input can be written as a weighted sum of shifted impulses, and the output is then a weighted sum of shifted impulse responses.

### Convolution
**Convolution** is the operation that gives the output of an LTI system as a combination of the input signal and the impulse response. The convolution of x(t) and h(t) is defined by an integral over all time of x(tau) multiplied by h(t - tau). Convolution is commutative, associative and distributive over addition. For example, convolving any signal with the unit impulse returns the same signal unchanged.

y(t) = x(t) * h(t) = integral of x(tau) h(t - tau) d tau

## Frequency Domain

### Fourier Series
The **Fourier series** represents a periodic signal as a sum of harmonically related sinusoids. The frequency of the lowest sinusoid is the **fundamental frequency**, and every other term is a multiple of it. The coefficients of the series tell how much of each harmonic is present in the signal.

### Fourier Transform
The **Fourier transform** extends the idea of the Fourier series to aperiodic signals and expresses a signal as a continuous mixture of complex exponentials. The Fourier transform converts a signal from the time domain to the frequency domain. A key property is that convolution in the time domain becomes multiplication in the frequency domain. The **frequency response** of an LTI system is the Fourier transform of its impulse response.

X(w) = integral of x(t) e^(-j w t) dt

### Sampling
**Sampling** is the process of converting a continuous-time signal into a discrete-time signal by measuring its value at regular intervals. The **sampling theorem** states that a band-limited signal can be reconstructed exactly if it is sampled at a rate greater than twice its highest frequency. This minimum rate is called the **Nyquist rate**. If the sampling rate is too low, different frequencies become indistinguishable, and this distortion is called **aliasing**. For example, a 5 kHz tone sampled at 8 kHz appears as a 3 kHz tone.

## Transforms for Analysis

### Laplace Transform
The **Laplace transform** generalises the Fourier transform by using complex frequency s instead of purely imaginary frequency. It converts differential equations in time into algebraic equations in s. The **region of convergence** is the set of values of s for which the Laplace transform integral converges. The stability of a causal LTI system can be read from the poles of its Laplace transform: the system is stable if all poles lie in the left half of the s-plane.

### Z-Transform
The **Z-transform** is the discrete-time counterpart of the Laplace transform. It converts difference equations into algebraic equations in z. For a causal discrete-time LTI system, the system is BIBO stable if all poles of the Z-transform lie inside the unit circle. Convolution of sequences in time becomes multiplication of their Z-transforms.
