## 因为源码的版本古老，其中牵涉到java语法的改变，所以不得不做一些修改
1. ant-1.3 org.apache.tools.ant.types.Path line:422 pos:26 将enum改为enu,以及接下来用到的enum改为enu
2. ant-1.3 org.apache.tools.ant.RuntimeConfigurable line: 141 pos: 26 将enum改为enu, 以及接下来用到的enum改为enu
3. ant-1.3 org.apache.tools.ant.Project 修复enum问题
4. ant-1.3 org.apache.tools.ant.taskdefs.AntStructure 修复enum问题