@[TOC]

对于Ubuntu来说，使用dpkg安装软件包是非常常用的一种操作。但是时间久了以后，我们可能会忘记当初安装了哪些软件包，以及如何安全的卸载软件包。在这一篇文章里，将告诉你这些平时不太注意到的小技巧。

# dpkg 安装软件包

除了使用 「apt」工具安装软件外，对于Debian/Ubuntu 这类的Linux系统，也可以使用「deb」包安装软件，例如：

```bash
$ sudo dpkg -i fping_4.2-1_amd64.deb
(正在读取数据库 ... 系统当前共安装有 252654 个文件和目录。)
准备解压 fping_4.2-1_amd64.deb  ...
正在解压 fping (4.2-1) 并覆盖 (4.2-1) ...
正在设置 fping (4.2-1) ...
正在处理用于 man-db (2.8.5-2) 的触发器 ...
```

# dpkg 查看已安装软件包

时间隔的久了后，可能会忘记安装了什么软件，此时可以执行下列命令：

```bash
$ dpkg -l
期望状态=未知(u)/安装(i)/删除(r)/清除(p)/保持(h)
| 状态=未安装(n)/已安装(i)/仅存配置(c)/仅解压缩(U)/配置失败(F)/不完全安装(H)/触发器等待(W)/触发器未决(T)
|/ 错误?=(无)/须重装(R) (状态，错误：大写=故障)
||/ 名称                                          版本                                体系结构     描述
+++-=============================================-===================================-============-===============================================================================
ii  2048-qt                                       0.1.6-1build1                       amd64        mathematics based puzzle game
ii  accountsservice                               0.6.50-0ubuntu1                     amd64        query and manipulate user account information
ii  acl                                           2.2.53-4                            amd64        access control list - utilities
ii  acpi-support                                  0.143                               amd64        scripts for handling many ACPI events
ii  acpid                                         1:2.0.31-1ubuntu2                   amd64        Advanced Configuration and Power Interface event daemon
ii  adduser                                       3.118ubuntu1                        all          add and remove users and groups
ii  adwaita-icon-theme                            3.32.0-1ubuntu1                     all          default icon theme of GNOME (small subset)
ii  alsa-base                                     1.0.25+dfsg-0ubuntu5                all          ALSA driver configuration files
ii  alsa-utils                                    1.1.8-1ubuntu1      
```

# dpkg 删除已安装软件包
如果有安装包需要卸载，我们可以执行以下指令：

```bash
$ sudo dpkg -r vim
(正在读取数据库 ... 系统当前共安装有 252653 个文件和目录。)
正在卸载 vim (2:8.1.0320-1ubuntu3.1) ...
update-alternatives: 使用 /usr/bin/vim.tiny 来在自动模式中提供 /usr/bin/vi (vi)
update-alternatives: 使用 /usr/bin/vim.tiny 来在自动模式中提供 /usr/bin/view (view)
update-alternatives: 使用 /usr/bin/vim.tiny 来在自动模式中提供 /usr/bin/ex (ex)
update-alternatives: 使用 /usr/bin/vim.tiny 来在自动模式中提供 /usr/bin/rview (rview)
```

# dpkg 查看软件包的安装位置

有时候我们可能需要查看软件包安装位置信息，可以执行下列指令：

```bash
$ dpkg -L fping
/.
/usr
/usr/bin
/usr/bin/fping
/usr/share
/usr/share/bug
/usr/share/bug/fping
/usr/share/doc
/usr/share/doc/fping
/usr/share/doc/fping/NEWS.Debian.gz
/usr/share/doc/fping/changelog.Debian.gz
/usr/share/doc/fping/copyright
/usr/share/lintian
/usr/share/lintian/overrides
/usr/share/lintian/overrides/fping
/usr/share/man
```

以上就是「dpkg」的最常用功能，如果有其他的需要，可以通过「dpkg --help」查看相关说明。