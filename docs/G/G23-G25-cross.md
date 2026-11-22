# 合并候选的右图换版追问

候选 `d568a564a26330f41b766fca0469a44a00fb96d8` 包含 G23、G24、G25。两个用例分别从空库打开 Chromium，导入脚本生成的双图 PDF；首轮询问第 1 页右图的横轴单位后，原位上传 v2。右图横跨页中线，左边界为 0.43，两个版本的右图裁区相同，图内单位从 Hz 变为 kHz。

| 换版后输入 | 观察结果 | 裁决 |
| --- | --- | --- |
| 跟上一版相比，第1页右图的横轴单位有什么变化？ | 进入 version_comparison，给出旧版 v1「[Hz]」和当前版 v2「kHz」。首轮和追问各发一次真实视觉请求，追问复用历史实读。 | 通过。 |
| 跟上一版比呢？ | 进入 version_comparison，提示本会话没有可核对的旧版引用。上一轮存在已显示且已实际读取的右图；本轮引用为空，新增视觉请求为零。 | 未通过。已识别跨版意图，未继承上一轮的视觉对象。 |

显式追问的旧版与当前版均对应 `page:1:image:2`。真实视觉请求图片、来源中的实际输入、点击引用后浏览器实际下载的图片 SHA-256 三方一致。v1 为 `dd7dce0a502f9cc126974b4e7e1324ed0b836dd974e4b59aeea5056632dd365a`，v2 为 `9ea5a6c56c32105e35c85e7739c35afae8a125b85a2a4edf0e3435070acfd37a`。两个引用打开第 1 页，版号与图片均正确。

从候选根目录运行，`SRAG_ENV` 指向带既有视觉配置的本机 .env。需要可用的 agent-browser、产品 Python 环境和视觉服务。

```sh
python docs/G/G23-G25-cross.py --env-file "$SRAG_ENV" --out /tmp/srag-explicit.json
python docs/G/G23-G25-cross.py --brief --env-file "$SRAG_ENV" --out /tmp/srag-brief.json
```

两种模式都先写入原始响应，再断言双版本来源、真实读图和引用图像一致。裸追问初测在修复前无法取得第二个字节摘要。上述数值来自本候选实测。

## 合并修订复审

重新启动两个互相独立的空库与 Chromium 会话。裸问“跟上一版比呢？”和完整指定右图横轴两组均进入 version_comparison，回答 v1 横轴单位 Hz、v2 横轴单位 kHz，已实际点击每个图像引用核对版号、页码及图片字节。每组首轮读取 v1，换版后只新增一次 v2 真实视觉请求。默认与 brief 模式的全部断言均通过。实测时 workspace_versions.py 的 SHA-256 为 `dd88266ab5979c94d0d9501702aa6aeffefe33b4555881203e7b87a558fabe17`。

[修订前后响应与实读摘要](./G23-G25-cross-results.json)

最终生产文件 SHA-256 为 58c50f8f8f5ff7467d5039c1f7ece6f2464ee4ed702c7afd359d8a3f9b194c73。整合者在该版本从两个空库重新运行上述显式与省略问句，两次均通过，两版的真实图像字节摘要与上表一致。
