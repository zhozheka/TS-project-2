import socket,struct,sys,time

Port=3004

def send_data(data):
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.connect(('localhost',Port))
    sock.sendall(struct.pack('>i', len(data))+data)
    data = recv_size(sock)
    sock.close()
    return data

def recv_size(the_socket):
    #data length is packed into 4 bytes
    total_len=0
    total_data=[]
    size=sys.maxsize
    size_data=b''
    sock_data=b''
    recv_size=4096
    while total_len<size:
        sock_data=the_socket.recv(recv_size)
        if len(sock_data) == 4:
            return ''
        if not total_data:
            if len(sock_data)>4:
                size_data+=sock_data
                size=struct.unpack('>i', size_data[:4])[0]
                recv_size=size
                if recv_size>524288:recv_size=524288
                total_data.append(size_data[4:])
            else:
                size_data+=sock_data
        else:
            total_data.append(sock_data)
        total_len=sum([len(i) for i in total_data ])
    return b''.join(total_data)

def start_server(processor):
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    sock.bind(('',Port))
    sock.listen(5)
    ender = None
    print ('started on',Port)
    while True:
        newsock,address=sock.accept()
        input_data=recv_size(newsock)

        try:
            answer = processor(input_data)
        except Exception as e:
            ender = e
            newsock.sendall(struct.pack('>i', 0))
            break
        newsock.sendall(struct.pack('>i', len(answer))+answer)
    print('processing ends')
    raise ender
