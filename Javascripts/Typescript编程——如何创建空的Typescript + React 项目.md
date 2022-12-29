@[TOC]

最近这两天在折腾Typescript，为了方便以后使用，写点一些浅显的东西，熟手请绕道。

# 什么是Typescript

TypeScript是一种由微软开发的强类型的JavaScript超集，可以在编译时进行类型检查。它在JavaScript的基础上增加了类型系统，并且可以使用JavaScript的所有功能。

TypeScript的目标是通过增加类型信息来提高代码的可读性和可维护性。它还支持面向对象编程的特性，如类和接口，以及静态类型检查。

TypeScript可以在任何JavaScript框架或库中使用，也可以作为独立的编程语言使用。它的语法与JavaScript非常相似，因此学习TypeScript需要具备JavaScript的基础知识。

使用TypeScript的优点包括：

**增强的代码可读性和可维护性**， TypeScript的类型系统可以帮助您在编写代码时更好地理解代码的意图，并且在更改代码时更容易发现错误。

**支持面向对象编程**，TypeScript支持面向对象编程的特性，如类和接口，使您可以使用这些特性来组织您的代码。

**可以捕获编译时的错误**，TypeScript在编译时进行类型检查，因此可以在运行时避免很多常见的错误。

# 常见的Typescript框架

常见的Typescript的框架有如下：

* **Angular**，是由Google开发的用于构建响应式单页应用的框架。它使用TypeScript作为默认的开发语言。
* **React**，是用于构建用户界面的JavaScript库。它可以与TypeScript一起使用，并且有一些工具（如create-react-app）可以帮助您轻松搭建TypeScript项目。
* **Vue.js**，是一个轻量级的JavaScript框架，用于构建单页应用。它可以与TypeScript一起使用，并提供了许多工具来帮助您在Vue.js项目中使用TypeScript。
* **Nest.js**，是一个基于Node.js的服务器端框架，使用TypeScript构建。它提供了一组有用的抽象层，可以帮助您快速构建服务器端应用程序。
* **Express**，是一个轻量级的Node.js服务器端框架。它可以与TypeScript一起使用，但并不提供与TypeScript相关的特性。

那么我们会想问一问，哪个框架目前使用人数最多？

在前端JavaScript框架中，React是目前使用人数最多的框架。根据2019年的调查数据，React占据了前端JavaScript框架市场的约35%，而Angular和Vue分别占据了约19%和18%的市场份额。

在服务器端JavaScript框架中，Express是目前使用人数最多的框架。根据2019年的调查数据，Express占据了服务器端JavaScript框架市场的约47%，而Nest占据了约8%的市场份额。

# 如何准备React项目
为了创建React项目，你需要先进行如下准备。

## 安装 Node 和 npm 包 
要准备使用React，您需要安装Node.js和npm（Node.js包管理器）。

首先，前往Node.js官网（https://nodejs.org/）下载并安装最新版本的Node.js。安装完成后，打开命令提示符或终端，输入以下命令以确认安装成功：

```bash
node -v
```
接下来，使用npm创建一个新的React项目。首先，创建一个新文件夹，然后在该文件夹中打开命令提示符或终端。输入以下命令以使用npm创建一个名为“my-app”的新项目：

```bash
npm init react-app my-app
```

然后

```bash
cd my-app
```

之后执行如下指令

```bash
npm start
```

不出意外，稍微等待几分钟，就可以访问 http://localhost:3000/ 可以看到刚刚创建的 React 项目。
