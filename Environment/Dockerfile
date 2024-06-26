# syntax=docker/dockerfile:1

FROM python:3.9-slim

WORKDIR /code/

RUN apt-get update 

# needs perl to run the command below
RUN apt-get install perl wget -y  

# fetch TinyTeX. If for any reason this were to fail, you can comment out
# lines 14-23, the code will execute without LaTeX formatting
RUN wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh 

# add TinyTeX to PATH variable
ENV PATH="$PATH:/root/.TinyTeX/bin/x86_64-linux"

# get LaTeX packages I need
RUN /root/.TinyTeX/bin/x86_64-linux/tlmgr install type1cm cm-super underscore 

#matplotlib wants dvipng as well (need to install it after tinytex to avoid some obscure version conflicts)
RUN apt-get install dvipng -y

# install python packages needed for my code
RUN pip install -U --no-cache-dir \
    numpy==1.25.2 \
    matplotlib==3.7.2 \
    pandas==1.5.3 \
    scipy==1.10.1 \
    contourpy==1.1.0 \
    cycler==0.11.0 \
    fonttools==4.42.0 \
    importlib-resources==6.0.1 \
    kiwisolver==1.4.4 \
    matplotlib-label-lines==0.6.0 \
    more-itertools==10.1.0 \
    packaging==23.1 \
    Pillow==10.0.0 \
    pyparsing==3.0.9 \
    python-dateutil==2.8.2 \
    pytz==2023.3 \
    six==1.16.0 \
    zipp==3.16.2


