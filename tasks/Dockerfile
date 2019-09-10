FROM nvidia/cuda:6.5-devel

# initial setup
RUN useradd -m -d /home/docker -s /bin/bash docker
RUN echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
#RUN chown docker:docker /home/docker

RUN apt-get -qq update
RUN apt-get install -y tzdata
RUN apt-get install -y gcc g++ git
RUN apt-get install -y aria2 curl wget bzip2
RUN apt-get install -y make unzip zlib1g-dev
RUN apt-get install -y sqlite openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev
RUN apt-get install -y tk-dev
RUN apt-get install -y emacs
RUN apt-get install -y graphviz
RUN apt-get install -y gfortran

# change user
USER docker
ENV HOME /home/docker
WORKDIR /home/docker
VOLUME /home/docker/tasks

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.profile
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
RUN pyenv install anaconda3-5.0.1
RUN pyenv global anaconda3-5.0.1
RUN pyenv init - >> ~/.profile
RUN pyenv init - >> ~/.bash_profile

# setup default python libraries
RUN conda update --all
RUN conda install scipy==1.2.1 pandas==0.23.4 numpy==1.16.2 scikit-learn
RUN conda install pytorch==1.1.0
RUN conda install -c rdkit rdkit==2018.03.4.0

# molgrammar

WORKDIR /home/docker
RUN git clone https://github.com/ibm-research-tokyo/graph_grammar.git
WORKDIR /home/docker/graph_grammar
RUN python setup.py install
WORKDIR /home/docker/graph_grammar/tasks
ENV MKL_THREADING_LAYER=GNU