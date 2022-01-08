// Authors: Florian Hofer
//
// License: BSD (3-clause)

#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION

#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *_squared_moving_integration(PyObject *self,
                                             PyObject *args,
                                             PyObject *kwargs)
{
    PyObject *input_array;
    int window_length;

    static char *kwarglist[] = {"x", "window_length", NULL};

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwargs,
                                     "Oi",
                                     kwarglist,
                                     &input_array,
                                     &window_length))
        return NULL;

    // ensure contiguous numpy-array
    input_array = PyArray_FROM_OTF(input_array,
                                   NPY_DOUBLE,
                                   NPY_ARRAY_C_CONTIGUOUS);

    if (input_array == NULL)
    {
        PyErr_SetString(PyExc_TypeError,
                        "_squared_moving_integration expected numeric array_like for x");
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject *)input_array) != 1)
    {
        PyErr_SetString(PyExc_ValueError,
                        "_squared_moving_integration only handles 1d-arrays!");
        Py_DecRef(input_array);
        return NULL;
    }

    npy_intp signal_len = PyArray_SHAPE((PyArrayObject *)input_array)[0];
    if (!(0 < window_length && window_length <= signal_len))
    {
        PyErr_SetString(
            PyExc_ValueError,
            "window_length has to be 0 < window_length <= len(x)");
        Py_DecRef(input_array);
        return NULL;
    }

    // get C-array containing data
    double *input = (double *)PyArray_DATA((PyArrayObject *)input_array);

    // create an output array of same length, type and memory-order
    PyObject *output_array = PyArray_NewLikeArray((PyArrayObject *)input_array,
                                                  NPY_ANYORDER,
                                                  NULL,
                                                  0);
    double *output = (double *)PyArray_DATA((PyArrayObject *)output_array);

    // create a circular buffer to store values inside integration window
    double *integration_buffer = (double *)calloc(window_length,
                                                  sizeof(double));

    double square;
    double sum = 0;

    // the integration window is centered on the original signal, for even
    // window_length the behavior of np.convolve with a constant window of
    // even length is replicated (i.e the window is off-center to the left)
    const int window_length_half = (window_length + 1) / 2;

    // during the first `window_length/2` samples there is no output, since
    // the integration window's center would be at a negative index of the
    // input
    for (int i = 0; i < window_length_half; ++i)
    {
        square = input[i] * input[i];
        integration_buffer[i % window_length] = square;
        sum += square;
    }

    for (int i = window_length_half; i < signal_len; ++i)
    {
        output[i - window_length_half] = sum; // write to 'window center'
        sum -= integration_buffer[i % window_length];
        square = input[i] * input[i];
        integration_buffer[i % window_length] = square;
        sum += square;
    }

    // the end of the input signal is reached, so the integration window is
    // built down and the last `window_length/2` entries of the output are
    // filled
    for (int i = (int)signal_len; i < signal_len + window_length_half; ++i)
    {
        output[i - window_length_half] = sum;
        sum -= integration_buffer[i % window_length];
    }

    free(integration_buffer);
    Py_DecRef(input_array);

    return output_array;
}

static PyObject *_thresholding(PyObject *self,
                               PyObject *args,
                               PyObject *kwargs)
{
    PyObject *filtered_ecg_array;
    PyObject *integrated_ecg_array;
    double fs;

    static char *kwarglist[] = {"filtered_ecg", "integrated_ecg", "fs", NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOd", kwarglist,
            &filtered_ecg_array, &integrated_ecg_array, &fs))
        return NULL;

    if (fs <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "fs has to be strictly positive");
        return NULL;
    }

    filtered_ecg_array = PyArray_FROM_OTF(filtered_ecg_array,
                                          NPY_DOUBLE,
                                          NPY_ARRAY_C_CONTIGUOUS);
    if (filtered_ecg_array == NULL)
    {
        PyErr_SetString(PyExc_TypeError,
                        "_thresholding expected numeric array_like for filtered_ecg_array");
        return NULL;
    }

    integrated_ecg_array = PyArray_FROM_OTF(integrated_ecg_array,
                                            NPY_DOUBLE,
                                            NPY_ARRAY_C_CONTIGUOUS);
    if (integrated_ecg_array == NULL)
    {
        PyErr_SetString(PyExc_TypeError,
                        "_thresholding expected numeric array_like for integrated_ecg_array");
        Py_DecRef(filtered_ecg_array);
        return NULL;
    }

    if (PyArray_NDIM((PyArrayObject *)filtered_ecg_array) != 1 ||
        PyArray_NDIM((PyArrayObject *)integrated_ecg_array) != 1)
    {
        PyErr_SetString(PyExc_ValueError,
                        "_thresholding only handles 1d-arrays!");
        Py_DecRef(filtered_ecg_array);
        Py_DecRef(integrated_ecg_array);
        return NULL;
    }

    double *filtered_ecg = (double *)PyArray_DATA((PyArrayObject *)filtered_ecg_array);
    double *integrated_ecg = (double *)PyArray_DATA((PyArrayObject *)integrated_ecg_array);

    npy_intp signal_len = PyArray_SHAPE((PyArrayObject *)integrated_ecg_array)[0];

    PyObject *beat_mask_array = PyArray_ZEROS(1,                                                  // ndim
                                              PyArray_SHAPE((PyArrayObject *)filtered_ecg_array), // shape
                                              NPY_BOOL,                                           // dtype
                                              0);                                                 // C-order
    char *beat_mask = (char *)PyArray_DATA((PyArrayObject *)beat_mask_array);

    int REFRACTORY_SAMPLES = (int)(0.2 * fs); // 200ms
    int T_WAVE_WINDOW = (int)(0.36 * fs);     // 360ms

    // --------------------------------------------------------------------
    // Learning Phase 1
    // --------------------------------------------------------------------
    // Pan & Tompkins mention a learning phase to initialize detection
    // thresholds based upon signal and noise peaks detected during the
    // first two seconds. The exact initialization process is not
    // described. The adaptive thresholds are calculated based on running
    // estimates of signal and noise peaks (`SPKF` and `NPKF` for the
    // filtered signal). Assuming constant peak amplitudes, those values
    // converge towards the signal peak amplitude and noise peak amplitude,
    // respectively. Therefore, SPKF/SPKI are assumed to be the maximum
    // values of the filtered/integrated signal during the learning phase.
    // Accordingly, NPKF/NPKI are initialized to the mean values during the
    // learning phase.
    unsigned int learning_phase_samples = (unsigned int)(2 * fs); // 2 seconds
    double filtered_ecg_maximum = 0;
    double filtered_ecg_sum = 0;
    double integrated_ecg_maximum = 0;
    double integrated_ecg_sum = 0;
    for (unsigned int i = 0; i < learning_phase_samples; ++i)
    {
        if (filtered_ecg[i] > filtered_ecg_maximum)
        {
            filtered_ecg_maximum = filtered_ecg[i];
        }
        if (integrated_ecg[i] > integrated_ecg_maximum)
        {
            integrated_ecg_maximum = integrated_ecg[i];
        }
        filtered_ecg_sum += filtered_ecg[i];
        integrated_ecg_sum += integrated_ecg[i];
    }

    double SPKF = filtered_ecg_maximum;
    double NPKF = filtered_ecg_sum / learning_phase_samples;
    double SPKI = integrated_ecg_maximum;
    double NPKI = integrated_ecg_sum / learning_phase_samples;
    double threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
    double threshold_F1 = NPKF + 0.25 * (SPKF - NPKF);

    // According to the original paper, `RR AVERAGE2` is the average of the
    // last 8 RR intervals that lie in a certain interval. In the worst
    // case, this requires going back to the very first RR interval.
    // Therefore, all RR intervals are stored. As the algorithm enforces a
    // refractory period, the maximum number of heartbeats is equal to
    // signal_len / refractory_samples.
    double *RR_intervals = (double *)calloc((int)(signal_len / REFRACTORY_SAMPLES),
                                            sizeof(double));

    // tracking the number of peaks found is required to calculate the
    // average RR intervals correctly during the first 8 beats (also needed
    // for access to RR_intervals)
    int num_peaks_found = 0;

    double RR_missed_limit;

    // in case a searchback was unsuccessful, no new searchback will be
    // performed until another signal peak has been found regularly
    char do_searchback = 1;

    // initialize, so searchback before any peak has been detected works
    int peak_index = -REFRACTORY_SAMPLES + 1;
    int previous_peak_index = -REFRACTORY_SAMPLES + 1;

    int index = 1;
    while (index < signal_len - 1)
    {
        double PEAKF;
        double PEAKI;

        char signal_peak_found = 0;
        char noise_peak_found = 0;
        // ----------------------------------------------------------------
        // Searchback
        // ----------------------------------------------------------------
        // During a "searchback", detection thresholds are reduced by one
        // half. The peak with highest amplitude between 200ms (i.e. the
        // refractory period) after the previous detected peak and the
        // current index is considered as a peak candidate.
        // Modifications compared to Pan & Tompkins' original method:
        // - The original paper states that a searchback peak's amplitude
        //   has to be between the original threshold and the reduced one.
        //   It can happen that this is the case for the filtered signal,
        //   but not for the integrated one (if the raw signal amplitude is
        //   suddenly considerably lower). Therefore, this implementation
        //   requires both signals (filtered and integrated) to be above
        //   the reduced threshold, but only one of them to be below the
        //   original threshold.
        // - No further steps are specified for the case that no peak is
        //   found during searchback. Since a searchback is triggered
        //   because (physiologically) there has to be a heartbeat during
        //   the searchback interval, this implementation repeats the
        //   process with further reduced thresholds. Up to 16 searchback
        //   runs are performed, each time the thresholds are further
        //   reduced by 1/2. A hard limit of 16 runs avoids an endless loop
        //   in case there's really just noise.
        // - Since the criterion for triggering a searchback is based on
        //   the average RR interval, in the original form this could only
        //   happen after at least two detected heartbeats. An
        //   exceptionally large peak during the first learning phase can
        //   throw the initial thresholds off, so peaks at the beginning
        //   are ignored - which in turn invalidates learning phase 2.
        //   Therefore, in addition to the original searchback criterion
        //   (no peak during 1.66 * "the average RR interval"), a
        //   searchback is triggered in two cases: (1) if there is no peak
        //   during the first second, and (2) if there is no peak 1.5s
        //   after the first peak.
        if ((num_peaks_found > 1 && index - previous_peak_index > RR_missed_limit && do_searchback) || // original criterion
            (num_peaks_found == 0 && index > fs) ||                                                    // (1)
            (num_peaks_found == 1 && index - previous_peak_index > 1.5 * fs))                          // (2)
        {
            for (int i = 1; i < 16; ++i)
            {
                char found_a_candidate = 0;

                int searchback_divisor = 1 << i; // 2^i
                int best_searchback_index = previous_peak_index + REFRACTORY_SAMPLES;
                double best_candidate_amplitude = -1;
                int searchback_index = best_searchback_index;

                while (searchback_index < index)
                {
                    PEAKF = filtered_ecg[searchback_index];
                    if (PEAKF > filtered_ecg[searchback_index + 1])
                    { // # next one is lower
                        if (PEAKF > filtered_ecg[searchback_index - 1])
                        { // # it is a peak
                            PEAKI = integrated_ecg[searchback_index];
                            // one signal is between the reduced and
                            // original threshold, the other one above the
                            // reduced threshold
                            if ((threshold_F1 / searchback_divisor < PEAKF && PEAKF < threshold_F1 && threshold_I1 / searchback_divisor < PEAKI) ||
                                (threshold_I1 / searchback_divisor < PEAKI && PEAKI < threshold_I1 && threshold_F1 / searchback_divisor < PEAKF))
                            {
                                if (PEAKF > best_candidate_amplitude)
                                { // highest one so far
                                    best_searchback_index = searchback_index;
                                    best_candidate_amplitude = filtered_ecg[searchback_index];
                                    found_a_candidate = 1;
                                }
                            }
                        }
                        // the amplitude of the next sample is lower, so it
                        // can't be a peak -> skip it
                        ++searchback_index;
                    }
                    ++searchback_index;
                }
                if (found_a_candidate)
                {
                    SPKI = 0.25 * PEAKI + 0.75 * SPKI;
                    SPKF = 0.25 * PEAKF + 0.75 * SPKF;
                    signal_peak_found = 1;
                    peak_index = best_searchback_index;

                    // don't perform a searchback until the next signal
                    // peak has been found to avoid endless loops
                    do_searchback = 0;
                    break;
                }
            }
        }
        else if (filtered_ecg[index] > filtered_ecg[index + 1])
        {
            if (filtered_ecg[index] > filtered_ecg[index - 1])
            {
                // a local maximum in the filtered signal was found
                PEAKF = filtered_ecg[index];
                PEAKI = integrated_ecg[index];
                if (PEAKF > threshold_F1 && PEAKI > threshold_I1)
                {
                    // Both the filtered and the integrated signal are
                    // above their respective thresholds. Thus the current
                    // peak is classified as a signal peak and the running
                    // estimates SPKF and SPKI are updated.
                    SPKF = 0.125 * PEAKF + 0.875 * SPKF;
                    SPKI = 0.125 * PEAKI + 0.875 * SPKI;

                    signal_peak_found = 1;
                    peak_index = index;
                }
                else
                {
                    noise_peak_found = 1;
                }
            }
            // The next sample's amplitude is lower, meaning it can't be a
            // peak, so we skip it. This is why there are two separate
            // if-clauses for this block.
            ++index;
        }

        // Calculating the RR interval and comparing slopes only makes
        // sense, if there has already been a signal peak in the past.
        if (signal_peak_found && num_peaks_found > 0)
        {
            double RR = peak_index - previous_peak_index;

            // ------------------------------------------------------------
            // T Wave Identification
            // ------------------------------------------------------------
            // "When an RR interval is less than 360 ms (it must be greater
            // than the 200 ms latency), a judgment is made to determine
            // whether the current QRS complex has been correctly
            // identified or whether it is really a T wave. If the maximal
            // slope that occurs during this waveform is less than half
            // that of the QRS waveform that preceded it, it is identified
            // to be a T wave; otherwise, it is called a QRS complex."
            // (from Pan & Tompkins, 1985)
            if (RR < T_WAVE_WINDOW)
            {
                int reverse_index = peak_index;
                double max_slope_in_this_peak = -1;
                while (reverse_index > 0)
                {
                    double amplitude_here = filtered_ecg[reverse_index];
                    double amplitude_before = filtered_ecg[reverse_index - 1];
                    if (amplitude_before > amplitude_here)
                        break;
                    double slope = amplitude_here - amplitude_before;
                    if (slope > max_slope_in_this_peak)
                        max_slope_in_this_peak = slope;
                    --reverse_index;
                }

                reverse_index = previous_peak_index;
                double max_slope_in_previous_peak = -1;
                while (reverse_index > 0)
                {
                    double amplitude_here = filtered_ecg[reverse_index];
                    double amplitude_before = filtered_ecg[reverse_index - 1];
                    if (amplitude_before > amplitude_here)
                        break;
                    double slope = amplitude_here - amplitude_before;
                    if (slope > max_slope_in_previous_peak)
                        max_slope_in_previous_peak = slope;
                    --reverse_index;
                }

                if (max_slope_in_this_peak < max_slope_in_previous_peak / 2.0)
                { // based on the slope, this peak should be a T Wave
                    signal_peak_found = 0;
                    noise_peak_found = 1;
                }
            }
        }

        if (signal_peak_found)
        {
            // What we know so far: we are at a local maximum, both
            // thresholds are exceeded and it is not a T wave. Thus, the
            // current sample can be considered as a "signal peak" and the
            // adaptive thresholds are updated.
            ++num_peaks_found;
            beat_mask[peak_index] = 1;

            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
            threshold_F1 = NPKF + 0.25 * (SPKF - NPKF);

            // calculating RR averages only makes sense once 2 peaks have
            // been found
            if (num_peaks_found > 1)
            {
                RR_intervals[num_peaks_found] = peak_index - previous_peak_index;

                double RR_low_limit;
                double RR_high_limit;

                // --------------------------------------------------------
                // Learning phase 2
                // --------------------------------------------------------
                // "Learning phase 2 requires two heartbeats to initialize
                // RR interval average and RR interval limit values."
                // (from Pan & Tompkins, 1985)
                if (num_peaks_found == 2)
                {
                    RR_low_limit = 0.92 * RR_intervals[num_peaks_found];
                    RR_high_limit = 1.16 * RR_intervals[num_peaks_found];
                }

                // --------------------------------------------------------
                // RR Average 1 / RR Average 2
                // --------------------------------------------------------
                // RR Average 2 is the average of the 8 most recent RR
                // intervals which fell between RR_low_limit and
                // RR_high_limit. In case of a regular heart rate, this
                // equals RR Average 1 (the average over the 8 most recent
                // RR intervals, independent of any limits). Therefore, RR
                // Average 1 does not need to be calculated separately.
                double RR_sum = 0;
                double RR_count = 0;
                char irregular = 0;
                for (int i = num_peaks_found; i > 1; --i)
                {
                    double RR_n = RR_intervals[i];
                    if (RR_low_limit < RR_n && RR_n < RR_high_limit)
                    {
                        RR_sum += RR_n;
                        ++RR_count;
                        if (RR_count >= 8)
                        {
                            break;
                        }
                    }
                    else
                    {
                        irregular = 1;
                    }
                }
                double RR_average = RR_sum / RR_count;

                RR_low_limit = 0.92 * RR_average;
                RR_high_limit = 1.16 * RR_average;
                RR_missed_limit = 1.66 * RR_average;

                if (irregular)
                {
                    // "For irregular heart rates, the first threshold of
                    // each set is reduced by half so as to increase the
                    // detection sensitivity and to avoid missing beats."
                    threshold_F1 /= 2;
                    threshold_I1 /= 2;
                }
            }

            // A signal peak has been found, so performing a searchback
            // makes sense.
            do_searchback = 1;

            // previous peak index is required to calculate the RR interval
            previous_peak_index = peak_index;

            // no peak can happen during the refractory period, so skip it
            index = peak_index + REFRACTORY_SAMPLES;
        }
        else if (noise_peak_found)
        {
            NPKI = 0.125 * PEAKI + 0.875 * NPKI;
            NPKF = 0.125 * PEAKF + 0.875 * NPKF;
            threshold_I1 = NPKI + 0.25 * (SPKI - NPKI);
            threshold_F1 = NPKF + 0.25 * (SPKF - NPKF);
        }

        ++index;
    }

    free(RR_intervals);

    Py_DecRef(filtered_ecg_array);
    Py_DecRef(integrated_ecg_array);

    // return an array containing `1`s at beat positions and `0`s elsewhere
    return beat_mask_array;
}

PyMethodDef method_table[] = {
    {"_squared_moving_integration",
     (PyCFunction)_squared_moving_integration,
     METH_VARARGS | METH_KEYWORDS,
     "Squares a signal and integrates in a sliding window."},

    {"_thresholding",
     (PyCFunction)_thresholding,
     METH_VARARGS | METH_KEYWORDS,
     "Modified version of Pan and Tompkin's thresholding algorithm"},

    {NULL, NULL, 0, NULL} // Sentinel value ending the table
};

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_heartbeat_detection", // Module name
    "efficient C-implementations of performance bottlenecks in heartbeat detection functions",
    -1,
    method_table,
};

PyMODINIT_FUNC PyInit__heartbeat_detection(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
