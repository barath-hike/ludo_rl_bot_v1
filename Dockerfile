FROM artifactory.hike.in:5001/ubuntu20_py3_9:v2
COPY ./pip.conf /root/.config/pip/
COPY ./pip.conf /etc/pip.conf
ADD ./ /online_learning
WORKDIR /online_learning
RUN /usr/local/bin/pip3.8 install --upgrade pip
RUN /usr/local/bin/pip3.8 install --upgrade --no-cache-dir -r requirements.txt