import wfdb
import subprocess

# A simple version of rdsamp for ease of use
# Return the physical signals and a few essential fields
def srdsamp(recordname, sampfrom=0, sampto=None, channels=None, pbdir=None):
    """Read a WFDB record and return the physical signal and a few important descriptor fields
    Usage:
    signals, fields = srdsamp(recordname, sampfrom=0, sampto=None, channels=None, pbdir=None)
    Input arguments:
    - recordname (required): The name of the WFDB record to be read (without any file extensions).
      If the argument contains any path delimiter characters, the argument will be interpreted as
      PATH/baserecord and the data files will be searched for in the local path.
    - sampfrom (default=0): The starting sample number to read for each channel.
    - sampto (default=None): The sample number at which to stop reading for each channel.
    - channels (default=all): Indices specifying the channel to be returned.
    Output arguments:
    - signals: A 2d numpy array storing the physical signals from the record.
    - fields: A dictionary specifying several key attributes of the read record:
        - fs: The sampling frequency of the record
        - units: The units for each channel
        - signame: The signal name for each channel
        - comments: Any comments written in the header
    Note: If a signal range or channel selection is specified when calling this function, the
          the resulting attributes of the returned object will be set to reflect the section
          of the record that is actually read, rather than necessarily what is in the header file.
          For example, if channels = [0, 1, 2] is specified when reading a 12 channel record, the
          'nsig' attribute will be 3, not 12.
    Note: The 'rdsamp' function is the base function upon which this one is built. It returns
          all attributes present, along with the signals, as attributes in a wfdb.Record object.
          The function, along with the returned data type, have more options than 'srdsamp' for
          users who wish to more directly manipulate WFDB files.
    Example Usage:
    import wfdb
    sig, fields = wfdb.srdsamp('sampledata/test01_00s', sampfrom=800, channels = [1,3])
    """
    print(1)
    record = wfdb.rdsamp(recordname, sampfrom, sampto, channels, True, pbdir, True)
    print(2)
    signals = record.p_signals
    fields = {}
    for field in ["fs", "units", "signame", "comments"]:
        fields[field] = getattr(record, field)

    return signals, fields


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    memory_used = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    memory_total = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.total',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Convert lines into a dictionary
    gpu_memory_used = [int(x) for x in memory_used.strip().split('\n')]
    gpu_memory_total = [int(x) for x in memory_total.strip().split('\n')]
    
    gpu_memory_map_used = dict(zip(range(len(gpu_memory_used)), gpu_memory_used))
    gpu_memory_map_total = dict(zip(range(len(gpu_memory_total)), gpu_memory_total))
    
    return gpu_memory_map_used, gpu_memory_map_total

