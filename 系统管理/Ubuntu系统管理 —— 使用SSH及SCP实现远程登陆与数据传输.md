@[toc]

# SSH

SSH 为 Secure Shell 的缩写，由 IETF 的网络小组（Network Working Group）所制定；SSH 为建立在应用层基础上的安全协议。SSH 是较可靠，专为远程登录会话和其他网络服务提供安全性的协议。利用 SSH 协议可以有效防止远程管理过程中的信息泄露问题。SSH最初是UNIX系统上的一个程序，后来又迅速扩展到其他操作平台。SSH在正确使用时可弥补网络中的漏洞。SSH客户端适用于多种平台。几乎所有UNIX平台—包括HP-UX、Linux、AIX、Solaris、Digital UNIX、Irix，以及其他平台，都可运行SSH[^1]。
 
[^1]: https://baike.baidu.com/item/ssh/10407?fr=aladdin

## SSH Server 的安装

在Windows平台，无论当作SSH服务器或客户端，一般情况都是需要依赖第三方工具，如果有需要假设服务器的朋友，可以参考这篇[文章](https://www.cnblogs.com/chenmingjun/p/8535067.html)。 而通常我们使用的SSH服务器，一般是针对Linux服务而言的。

我平时用Debian、Ubuntu这类系统比较多，所以对于Ubuntu来说，安装SSH服务，可以执行下述命令：

```bash
$ sudo apt install openssh-server
```

对于现在Ubuntu 18.04以上版本来说，已经不需要在安装后编辑相关配置文件，但是如果需要禁止root权限登陆的话，那么需要在 **/etc/ssh/sshd_config** 中把 「**PermitRootLogin**」 从NO改为YES即可，而至于其他，比如修改默认的22端口，如果没什么特别的理由，这样会导致很多基于22端口通信的服务出现异常，比如GIT。

## 通过指定端口连接远程服务

通常Linux系统都是作为服务器，而大多数相信看我文章的朋友，大概使用云服务器的机会多过真物理机的机会。有时候因为路由器或者交换机端口映射的原因，又或者为了绕开公网对22端口的限制，或者虚拟专用网络的原因，你能使用到的服务器通常SSH的可访问是其他的端口号，比如1121，3348等。

而如果执行

```bash
$ ssh user@remote.server.ip.or.name.com
```

多会提示远程服务无法访问，所以这个时候我们就可以通过「-p」这个指令，显式的指定访问端口，例如：

```bash
$ ssh user@remote.server.ip.or.name.com -p 1128
```

## 远程免密登陆

远程免密登陆，通常是一种比较省事的操作，因为每次执行ssh登陆指令，都会被要求输入密码，所以对于懒人来说，我们可以通过「ssh-copy-id -i」这个指令，把本地计算机的公钥部署到SSH服务器上，以此达成免密登陆。

至于怎么做呢，首先

### 生成公钥
我们在本地，首先执行

```bash
$ ssh-keygen -t rsa
```
生成用RSA加密的公私钥。

### 把公钥发送给SSH服务器

加密钥匙生成完毕后，我们需要把公钥发送给SSH服务器，主要执行下述指令

```bash
$ ssh-copy-id -i ~/.ssh/id_rsa.pub  user@remote.server.ip.or.name.com
```

输入SSH登陆密码后，就可以免密正常登陆了。


# SCP

Linux scp 命令用于 Linux 之间复制文件和目录。scp 是 secure copy 的缩写, scp 是 linux 系统下基于 ssh 登陆进行安全的远程文件拷贝命令。scp 是加密的，rcp 是不加密的，scp 是 rcp 的加强版[^2]。

[^2]: https://www.runoob.com/linux/linux-comm-scp.html

## 将数据从本地传输至远程服务器

数据发送到远程设备上，一般有FTP、HTTP等协议，而基于SSH协议传输数据的SCP命令也是我们可以使用的重要工具。

```bash
scp local_file remote_username@remote_ip:remote_folder 
或者 
scp local_file remote_username@remote_ip:remote_file 
或者 
scp local_file remote_ip:remote_folder 
或者 
scp local_file remote_ip:remote_file 
```

如果需要一次性把一个文件夹的数据都发送给远程服务器，就需要执行下述指令：

```bash
scp -r local_folder remote_username@remote_ip:remote_folder 
或者 
scp -r local_folder remote_ip:remote_folder 
```

比如，我们要把本地的代码，上传给服务器，然后在服务器上运行或者部署，就可以这样
```bash
$ scp -r ~/Desktop/SourceDir user@remote.server.ip.or.name.com:~/Desktop
```

## 将数据从远程服务器拉取到本地

scp除了可以把数据从本地上传到远程服务器，也可以反过来，把远程服务器的文件拉取到本地。

```bash
$ scp -r user@remote.server.ip.or.name.com:~/Desktop/Dir  ~/Downloads
```

「-r」这个指令表示递归，如果只是拉取一个文件，就不需要了。

## 指定端口方法

同样，我们也存在着远程服务器SSH端口不在22号的问题，所以需要使用「-p」，指定远程设备的端口号。这样当我们希望把数据推送到远程时，就可以这样执行：

```bash
$ scp -p 1121 -r ~/Desktop/Data/   user@remote.server.ip.or.name.com:~/Desktop
```

执行拉取命令也是一样的

```bash
$ scp -p 1121 -r user@remote.server.ip.or.name.com:~/Desktop  ./
```