@[toc]

# SD卡初始化

这一步非必要，如果SD卡之前已经做过系统的，你如果担心烧制过程中出现意外情况，可以执行以下步骤。首先需要通过 **fdisk -l** 查看SD卡是否正常被系统识别到，如果没有识别到就需要把SD卡拔出来，然后换个别的接口插回去。

```bash
$ sudo fdisk -l
Disk /dev/sdc: 29.8 GiB, 32010928128 bytes, 62521344 sectors
Disk model: Card  Reader    
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disklabel type: dos
Disk identifier: 0x1a73246f

Device     Boot Start      End  Sectors  Size Id Type
/dev/sdc1        2048 62521343 62519296 29.8G 83 Linux
```

确定某个设备号所对应的信息都是对应的 SD 卡后，就可以执行 **fdisk** 命令，执行删除分区信息等操作。


```bash
$ sudo fdisk /dev/sdc1
Welcome to fdisk (util-linux 2.33.1).
Changes will remain in memory only, until you decide to write them.
Be careful before using the write command.

Device does not contain a recognized partition table.
Created a new DOS disklabel with disk identifier 0x83ea5e8c.

Command (m for help): 
```

先执行 **d** 删除全部分区，再执行 **w** 修改文件系统信息，然后我们就可以直接准备烧录镜像文件了。

# 卸载 SD 卡
这里的卸载指的是我们要先用 **umount** 命令把SD卡从挂载设备信息上卸载掉，这样才能写入镜像文件。

```bash
$ sudo umount /dev/sdc
umount: /dev/sdc: not mounted.
```

这里的 **/dev/sdc** 是我的 SD 卡挂载文件符，你自己的可能是 /dev/sdb 或者其他，反正注意看，成功后就可以看到上面的输出信息了。

# 烧录镜像文件
然后是烧录镜像文件，我们要使用到 **dd** 指令。在执行前，我们先 **cd** 到下载好的镜像文件目录。因为我的镜像文件存放在 ~/Downloads 目录下，所以：

```bash
$ cd ~/Downloads
```

如果下载的文件是 zip 包，你需要先解压后再烧录，如果已经是 iso 或者 img 等镜像文件格式，可以直接执行以下命令

```bash
$ sudo dd if=./2021-10-30-raspios-bullseye-armhf-full.img of=/dev/sdc bs=16MB
```
if 是镜像，of 是烧录地址，bs 是同时读取和写入字节数，然后等待一段时间后就可以了。

