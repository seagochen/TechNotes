@[toc]

# 硬盘初始化

首先需要通过 **fdisk -l** 查看硬盘是否正常被系统识别到，如果没有识别到就需要把硬盘拔下来，然后换个接口插回去。

```bash
pi@raspberrypi:~ $ sudo fdisk -l
Disk /dev/sda: 115.5 PiB, 129986248068418560 bytes, 253879390758630 sectors
Disk model: USB3.0 DISK00
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
Disklabel type: dos
Disk identifier: 0x35614444

Device     Boot Start        End    Sectors  Size Id Type
/dev/sda1        2048 3907028991 3907026944  1.8T 83 Linux


Disk /dev/sdb: 115.5 PiB, 129986248068418560 bytes, 253879390758630 sectors
Disk model: USB3.0 DISK01
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 4096 bytes / 4096 bytes
Disklabel type: dos
Disk identifier: 0xdca5237a

Device     Boot Start        End    Sectors  Size Id Type
/dev/sdb1        2048 3906252799 3906250752  1.8T 83 Linux
```

在我的RaspberryPi系统上目前一共挂载了两块硬盘，识别出的标识符分别是

~~~bash
/dev/sda
/dev/sdb
~~~

# 硬盘初始化和格式化

硬盘的初始化也是通过 **fdisk** 工具实现的，例如：

~~~bash
pi@raspberrypi:~ $ sudo fdisk /dev/sda

Welcome to fdisk (util-linux 2.33.1).
Changes will remain in memory only, until you decide to write them.
Be careful before using the write command.

The size of this disk is 115.5 PiB (129986248068418560 bytes). DOS partition table format cannot be used on drives for volumes larger than 4294966784 bytes for 512-byte sectors. Use GUID partition table format (GPT).

Command (m for help):
~~~

此时可以执行的命令，常用的有几个:

* **p** 打印当前的硬盘分区信息
* **d** 删除既有的硬盘分区信息
* **w** 写入新的硬盘分区信息
* **n** 创建新的硬盘分区信息
* **q** 退出工具


如果要删除既有的硬盘分区，然后再创建新的分区，执行过程为： d $\rightarrow$ n $\rightarrow$ w $\rightarrow$ q。

如果是新硬盘，那么直接执行 n $\rightarrow$ w $\rightarrow$ q 就行了。


# 常见错误

有时候，由于硬盘被系统挂载，会提示硬盘分区信息写入错误，这个就需要通过 **df** 或者
 **df -h** 查看硬盘挂载信息

~~~bash
pi@raspberrypi:~ $ df
Filesystem      1K-blocks      Used  Available Use% Mounted on
/dev/root        26817268   8364564   17067412  33% /
devtmpfs           439920         0     439920   0% /dev
tmpfs              473200         0     473200   0% /dev/shm
tmpfs              473200      6564     466636   2% /run
tmpfs                5120         4       5116   1% /run/lock
tmpfs              473200         0     473200   0% /sys/fs/cgroup
/dev/mmcblk0p6     258094     48789     209306  19% /boot
/dev/sdb1      1921420504 614166596 1209581256  34% /media/pi/Data
/dev/sda1      1921802432 117623708 1706486668   7% /media/pi/Repositories
tmpfs               94640         4      94636   1% /run/user/1000
overlay          26817268   8364564   17067412  33% /var/lib/docker/overlay2/6a589f1ac224f9653ea0c1010fc2d93648efa30a641c3828f12b44779e024b9e/merged
overlay          26817268   8364564   17067412  33% /var/lib/docker/overlay2/40877c69487c2da3d0bcf4b2b35188c65229a5960955ca694f5d1f932018993f/merged
~~~

找到疑是挂载点，执行 **umount** 命令，把硬盘从挂载点上退出后，再执行上面的步骤。


# 格式化硬盘

硬盘分区信息写完后，就需要对分区进行格式化，这一步的逻辑和Windows是一致的。可以使用 **mkfs.ext4** 把硬盘以Linux目前常用的ext4格式进行格式化，如果硬盘需要和Windows共用，建议使用 **mkfs.fat** 或 **mkfs.exfat** 把分区格式化为FAT格式或者exFAT格式，这样在不同的系统下就可以读取数据了。

命令执行方式大致如下：

~~~bash
 sudo mkfs.ext4 /dev/sda1
~~~


# 自动挂载

硬盘格式化成功后，可以通过 **mount** 命令，把硬盘挂载起来使用，不过这不是一个好办法，如果你希望每次系统启动后，都能自动挂载硬盘，那么需要编写 /etc/fstab 文件。

在文件末尾，加入这样一行命令：

~~~bash
	# 硬盘分区   挂载点          分区文件类型     设置       dump   fsck
	/dev/sda1   /media/pi/sda1   ext4            defaults   0      1
~~~


然后重启后，就能看到硬盘正常挂载和使用了。
