FROM ubuntu:16.04

ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p vectors/fasttext

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         curl \
         bzip2 \
         git \
         unzip \
         make \
         g++ \
         gzip \
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# download bert repo
RUN git clone https://github.com/google-research/bert.git

# clone semeval subtask A data
RUN git clone https://github.com/Semeval2019Task9/Subtask-A.git

# clone semeval subtask B data
RUN git clone https://github.com/Semeval2019Task9/Subtask-B.git