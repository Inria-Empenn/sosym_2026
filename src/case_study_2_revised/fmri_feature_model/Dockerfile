FROM python:3.11

ARG MATLAB_VERSION=R2024b
ARG AGREE_TO_MATLAB_RUNTIME_LICENSE=yes
ARG SPM_VERSION=25
ARG SPM_RELEASE=25.01
ARG SPM_REVISION=02
# Calculated
ENV SPM_TAG=${SPM_RELEASE}${SPM_REVISION:+.${SPM_REVISION}}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install unzip xorg wget
RUN apt-get clean
RUN rm -rf \
    /tmp/hsperfdata* \
    /var/*/apt/*/partial \
    /var/lib/apt/lists/* \
    /var/log/apt/term*

ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/${MATLAB_VERSION}/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/${MATLAB_VERSION}/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/${MATLAB_VERSION}/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/${MATLAB_VERSION}/sys/opengl/lib/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/${MATLAB_VERSION}/extern/bin/glnxa64
ENV MCR_INHIBIT_CTF_LOCK=1
ENV SPM_HTML_BROWSER=0

RUN echo "ulimit -n 65536" >> /root/.bashrc

# Install SPM Standalone in /opt/spm/
# Running SPM once with "function exit" tests the succesfull installation *and*
# extracts the ctf archive which is necessary if singularity is going to be
# used later on, because singularity containers are read-only.
# Also, set +x on the entrypoint for non-root container invocations
RUN wget --no-check-certificate --progress=bar:force -P /opt https://github.com/spm/spm/releases/download/${SPM_TAG}/spm_standalone_${SPM_TAG}_Linux.zip
RUN unzip -q /opt/spm_standalone_${SPM_TAG}_Linux.zip -d /opt
RUN rm -f /opt/spm_standalone_${SPM_TAG}_Linux.zip
RUN mv /opt/spm_standalone /opt/spm
RUN /opt/runtime_installer/Runtime_${MATLAB_VERSION}_for_spm_standalone_${SPM_TAG}.install -agreeToLicense ${AGREE_TO_MATLAB_RUNTIME_LICENSE}
RUN ulimit -n 65536 && /opt/spm/spm${SPM_VERSION} function exit
RUN chmod +x /opt/spm/spm${SPM_VERSION}
RUN ln -s /opt/spm/spm${SPM_VERSION} /usr/local/bin/spm

WORKDIR /app
COPY . /app
COPY ./run_spm25.sh /opt/spm/run_spm25.sh
RUN chmod +x /opt/spm/run_spm25.sh

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "run.py", ""]
