#Instructions

The command line syntax for executing the IDA Download jar file:

    java -jar IdaDownloader_XXXX.jar --directory=YYY --chunks=10 <URL>

Where:

    * --directory is the local directory to write the download to
    * --chunks (optional) is the initial number of chunks that the download is divided into (between 1 and 20)
    * <URL> is the URL for the download obtained from the IDA download web page

Java version 12 or higher is required.
