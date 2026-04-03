% myclass.cls

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myclass}[2024/01/01 v1.0 My Custom LaTeX Class]

% 1. 加载基础类 (使用 article，纸张默认为 A4)
\LoadClass[a4paper, zihao=-4]{ctexart}
\setlength{\parindent}{2em} % 设置全局段落缩进为2em
% 2. 加载页面布局宏包
\RequirePackage{unicode-math} % 数学字体支持
\RequirePackage{amsmath}  % 数学公式支持
% \RequirePackage{amssymb}  % 数学符号支持 和 unicode-math 冲突，按需选择
\RequirePackage{tabularx}  % 表格支持
\RequirePackage{geometry}  % 页面布局
\RequirePackage{graphicx}  % 图片支持
\RequirePackage{tikz}      % 矢量示意图支持
\RequirePackage{setspace}  % 行距控制
\RequirePackage{fontspec}  % 字体设置
\RequirePackage{fancyhdr} % 标题格式控制
\RequirePackage{xurl}     % 【新增】允许URL在任意位置自动断行
\RequirePackage{hyperref} % 超链接支持
\RequirePackage{titletoc}  % 用于自定义目录格式
\RequirePackage{titlesec}  % 用于自定义章节标题格式
\RequirePackage{booktabs}  % 用于自定义章节标题格式
\RequirePackage{caption}    % 【新增】用于控制标题格式
\RequirePackage{etoolbox}   % 【新增】用于自动化设置表格内容字体
\RequirePackage[super]{gbt7714} % 参考文献支持（内部已加载natbib）
\usetikzlibrary{arrows.meta,calc,positioning}
% 3. 配置英文主字体为 Times New Roman
\setmainfont{Times New Roman}

% 4. 配置页面参数 (保持你之前的要求)
\geometry{
    a4paper,            % 纸张大小
    top=3cm,            % 上边距 3cm
    bottom=2.5cm,       % 下边距 2.5cm
    left=2.5cm,         % 左边距 2.5cm
    right=2.5cm,        % 右边距 2.5cm
    bindingoffset=1cm,  % 装订线 1cm (靠左，即奇数页左侧增加1cm)
    % --- 页眉页脚位置计算 ---
    % 说明：LaTeX 的 top 是指正文上边缘到纸张上边缘的距离。
    % 若要在视觉上保证页眉位于 1.6cm 处，我们需要调整 headsep。
    % 假设页眉高度(headheight)为 0.5cm (约14pt)，
    % headsep = top - 页眉位置 - headheight = 3.0 - 1.6 - 0.5 = 0.9cm
    headheight=0.5cm,
    headsep=0.9cm,
    % 同理，footskip 控制页脚基线到正文下边缘的距离。
    % 若要页脚位于 1.5cm 处：
    % footskip ≈ bottom - 页脚位置 = 2.5 - 1.5 = 1.0cm
    footskip=1.0cm
}

% 5. 创建专门用于摘要页的页眉样式
\fancypagestyle{cnabstractpage}{
    \fancyhf{} % 清空所有页眉页脚
    \fancyhead[L]{\songti\zihao{-5} 重庆大学本科学生毕业论文（设计）}
    \fancyhead[R]{\songti\zihao{-5} 摘要}
    \renewcommand{\headrulewidth}{0.4pt} % 页眉横线
    \renewcommand{\footrulewidth}{0pt}   % 取消页脚横线
    \fancyfoot[C]{\thepage} % 页码居中
}

% 6. 定义普通页面的页眉样式
\fancypagestyle{fancy}{
    \fancyhf{} % 清空当前设置
    \fancyhead[L]{\songti\zihao{-5} 重庆大学本科学生毕业论文（设计）}

    % 关键修改：这里用 \leftmark 代替了具体的文字
    \fancyhead[R]{\songti\zihao{-5} \leftmark}

    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancyfoot[C]{\thepage}
}

\renewcommand{\sectionmark}[1]{%
    % \markboth{左标记内容}{右标记内容}
    % \thesection 是序号(如 1), #1 是标题内容(如 绪论)
    \markboth{\thesection\quad #1}{}
}

% 7. 设置全局页面样式
\pagestyle{fancy}

% 定义封面命令 \makecover
\newcommand{\makecover}{
    \begin{titlepage}
        \centering
        \singlespacing

        % --- 1. 学校名称 ---
        \vspace*{0pt}

        { \heiti \zihao{-2} 重庆大学本科学生毕业论文（设计） \par}

        % --- 空两行 (小二) ---
        {\zihao{-2} \setlength{\baselineskip}{35pt}  \vspace{2\baselineskip}}

        % --- 2. 论文题目 ---
        { \heiti \zihao{2} 风电功率深度学习预测方法研究 \par}

        % --- 空三行 (五号) ---
        { \zihao{5} \setlength{\baselineskip}{18pt} \vspace{3\baselineskip}}

        % --- 3. 图片 ---
        \includegraphics[width=4.27cm, height=4.19cm]{figure/logo.png}

        % --- 空两行 (三号) ---
        { \zihao{3} \setlength{\baselineskip}{21.5pt} \vspace{2\baselineskip}}

        % --- 4. 学生信息部分 (重点修改) ---
        % 设置左对齐容器，消除居中影响，以便进行精确缩进
        % 字体：黑体三号
        % 行距：单倍
        {
            \heiti \zihao{3} % 设定字体
            %设置行高。
            \setlength{\baselineskip}{31pt}

            \begin{flushleft}
                % 设置右缩进 0.03cm
                \setlength{\rightskip}{0.03cm}
                % 设置左缩进 3.49cm
                \setlength{\leftskip}{3.49cm}

                % [s] 表示分散对齐
                \newcommand{\fieldlabel}[1]{\makebox[4em][s]{##1}}

                % 逐行输出
                \noindent \fieldlabel{学生}：王锦源 \par
                \noindent \fieldlabel{学号}：20221067 \par
                \noindent \fieldlabel{指导教师}：赵霞 \par
                \noindent \fieldlabel{专业}：电气工程及其自动化 \par
            \end{flushleft}
        }

        % --- 空两行 (三号) ---
        % 由于上面用了 flushleft，这里需要恢复居中，或者直接用 center 环境
        % 注意：为了保证垂直间距准确，我们退出 flushleft 后加间距
        {\zihao{3}  \setlength{\baselineskip}{31pt} \vspace{2\baselineskip}}

        % --- 5. 学院名称 ---
        % 小二，居中
        {\centering \heiti \zihao{-2} 重庆大学电气工程学院 \par}

        % --- 6. 日期 ---
        % 黑体三号，段前12磅，居中
        \vspace{21pt}
        {\centering \heiti \zihao{3} 2026年6月 \par}

    \end{titlepage}
}

% 定义英文封面命令 \Englishmakecover
\newcommand{\Englishmakecover}{
    \begin{titlepage}
        \centering
        \singlespacing

        % --- 1. 学校名称 / Header ---
        \vspace*{-7pt}

        % 【修改】使用 \rmfamily (Times New Roman) + \bfseries (加粗)
        { \rmfamily \bfseries \zihao{-3} Undergraduate Thesis (Design) of Chongqing University \par}

        % --- 空两行 (小二) ---
        {\zihao{-2} \setlength{\baselineskip}{35pt}  \vspace{2\baselineskip}}

        % --- 2. 论文题目 / Title ---
        % 【修改】字体改为 Times New Roman 加粗
        { \rmfamily \bfseries \zihao{2} Research on Deep Learning Prediction Method for Wind Power \par}

        % --- 空两行 (五号) ---
        { \zihao{5} \setlength{\baselineskip}{14pt} \vspace{2\baselineskip}}

        % --- 3. 图片 / Logo ---
        \includegraphics[width=4.27cm, height=4.19cm]{figure/logo.png}

        % --- 空两行 (三号) ---
        { \zihao{3} \setlength{\baselineskip}{21.5pt} \vspace{2\baselineskip}}

        % --- 学生个人信息 ---
        { \rmfamily \bfseries \zihao{3} By \vspace{16pt}}

        { \rmfamily \bfseries \zihao{3} WANG Jin Yuan }

        { \zihao{3} \setlength{\baselineskip}{32pt} \vspace{1\baselineskip} }

        { \rmfamily \bfseries \zihao{3} Supervised by \vspace{17pt}}

        { \rmfamily \bfseries \zihao{3} Prof. ZHAO Xia \vspace{16pt} }

        { \zihao{3} \setlength{\baselineskip}{33pt} \vspace{1\baselineskip} }

        % 专业    
        { \rmfamily \bfseries \zihao{-2} Electrical Engineering and Automation \vspace{17pt}}

        % 学院    
        { \rmfamily \bfseries \zihao{-2} SCHOOL OF ELECTRICAL ENGINEERING \vspace{17pt}}

        % --- Chongqing University 前间距（已缩小） ---


        % --- 5. 学院名称 / School Name ---
        % 【修改】字体改为 Times New Roman 加粗
        {\centering \rmfamily \bfseries \zihao{-2} Chongqing University \vspace{17pt}}

        % --- 6. 日期 / Date ---

        % 【修改】字体改为 Times New Roman 加粗
        { \centering \rmfamily \bfseries \zihao{3} June, 2026 }

    \end{titlepage}
}

% 定义中文摘要环境 cnabstract
\newenvironment{cnabstract}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{section}{摘要}

    % 设置当前页面样式为摘要页样式
    \thispagestyle{cnabstractpage}

    % --- 标题部分 ---
    \vspace*{-13pt}

    \begin{center}
        \heiti \zihao{3} 摘\quad 要
    \end{center}

    \vspace{8pt}

    % --- 正文设置 ---
    \songti \zihao{-4}
    \setlength{\baselineskip}{20pt}
    \parindent 2em
}{
    \par
}

% 定义关键词命令 \cnkeywords
\newcommand{\cnkeywords}[1]{
    % 换行并留出间距
    \par \vspace{20pt}
    {
        \setlength{\leftskip}{2.2em} % ← 这里调整整行的左缩进
        \noindent
        {\heiti \zihao{-4} 关键词：}
        {\songti \zihao{-4} #1}
        \par
    }
    % 确保关键词后正确分页
    \clearpage
}

% 定义英文摘要环境 enabstract
\newenvironment{enabstract}{
    \clearpage
    \phantomsection
    \addcontentsline{toc}{section}{ABSTRACT}

    % 设置当前页面样式为摘要页样式
    \thispagestyle{enabstractpage} % 需要定义 enabstractpage 样式

    % --- 标题部分 ---
    \vspace*{-13pt}

    \begin{center}
        \rmfamily \bfseries \zihao{3} ABSTRACT
    \end{center}

    \vspace{8pt}

    % --- 正文设置 ---
    \rmfamily \zihao{-4} % Times New Roman 小四
    \setlength{\baselineskip}{20pt}
    \parindent 2em
}{
    \par
}

% 定义英文关键词命令 \enkeywords
\newcommand{\enkeywords}[1]{
    % 换行并留出间距
    \par \vspace{20pt}
    {
        \setlength{\leftskip}{0em} % 与中文保持相同缩进
        \noindent
        {\rmfamily \bfseries \zihao{-4} Keywords: } % 加粗
        {\rmfamily \zihao{-4} #1} % 不加粗
        \par
    }

    % 确保关键词后正确分页
    \clearpage
}

% 在类文件中添加英文摘要页样式（如果还没有的话）
\fancypagestyle{enabstractpage}{
    \fancyhf{} % 清空所有页眉页脚
    \fancyhead[L]{\songti\zihao{-5} 重庆大学本科学生毕业论文（设计）}
    \fancyhead[R]{\zihao{-5} ABSTRACT}
    \renewcommand{\headrulewidth}{0.4pt} % 页眉横线
    \renewcommand{\footrulewidth}{0pt}   % 取消页脚横线
    \fancyfoot[C]{\thepage} % 页码居中
}

\renewcommand{\thesection}{\arabic{section}}

% 二级标题编号：1.1
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}

% 三级标题编号：1.1.1
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}

% 设置章节标题格式

\newcommand{\sectionbreak}{\clearpage}
\titleformat{\section}%一级标题另起一页
{\vspace*{-5.2pt}\centering\heiti\fontspec{SimHei}\zihao{3}}{\thesection}{0.6em}{}
%编号和标题之间的距离为0.5em，顶部距离，强制英文黑体

\titleformat{\subsection}
{\heiti\fontspec{SimHei}\zihao{-3} }{\thesubsection}{0.6em}{}

\titleformat{\subsubsection}
{\heiti\fontspec{SimHei}\zihao{-3} }{\thesubsubsection}{0.6em}{}

% --- 设置二级标题 (subsection) 的间距 ---
% \titlespacing{命令}{左缩进}{段前间距}{段后间距}
\titlespacing{\subsection}{0pt}{13pt}{0pt}
%2级标题换行

% --- 设置三级标题 (subsubsection) 的间距 ---
\titlespacing{\subsubsection}{0pt}{0pt}{0pt}

% 定义目录的页眉样式
\fancypagestyle{tablecontents}{
    \fancyhf{}
    \setlength{\headheight}{11pt}    % 调试页眉内容区域高度
    \setlength{\headsep}{51pt}       % 调试页眉与正文间距
    \fancyhead[L]{\songti\zihao{-5} 重庆大学本科学生毕业论文（设计）}
    \fancyhead[R]{\songti\zihao{-5} 目录}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancyfoot[C]{\thepage}
}

% 设置目录格式
\renewcommand{\tableofcontents}{%
    \clearpage
    \phantomsection
    % 添加目录到目录中
    % \addcontentsline{toc}{section}{\contentsname}

    % 设置目录页页眉样式
    \thispagestyle{tablecontents} % 使用和摘要页相同的样式
    \fancyhead[R]{\songti\zihao{-5} 目录} % 修改右侧为"目录"

    % 目录标题：黑体三号居中
    \begin{center}
        \heiti \bfseries \zihao{3} 目\quad 录
    \end{center}

    \vspace{7pt} % 与摘要保持相同的间距

    % 开始生成目录
    \@starttoc{toc}

    % 确保目录后正确分页
    \clearpage
}

% 设置目录条目格式
% 一级标题（section）：黑体4号
\titlecontents{section}
[2em] % 左缩进
{\addvspace{0pt}\heiti\zihao{4}\setlength{\baselineskip}{20pt}} % 格式
{\contentslabel{1.5em}} % 带编号的标签
{\hspace*{-1.5em}} % 无编号的标签
{\titlerule*[0.4em]{.}\contentspage} % 页码和引导符

% 二级标题（subsection）：黑体5号
\titlecontents{subsection}
[2.7em] % 左缩进（比一级标题缩进2em）
{\addvspace{0pt}\heiti\zihao{5}\setlength{\baselineskip}{20pt}} % 格式
{\contentslabel{2em}} % 带编号的标签
{\hspace*{-2.7em}} % 无编号的标签
{\titlerule*[0.6em]{.}\contentspage} % 页码和引导符

% 三级标题（subsubsection）：宋体5号
\titlecontents{subsubsection}
[4em] % 左缩进（比二级标题再缩进2em）
{\addvspace{0pt}\songti\zihao{5}\setlength{\baselineskip}{20pt}} % 格式
{\contentslabel{2.5em}} % 带编号的标签
{\hspace*{-4em}} % 无编号的标签
{\titlerule*[0.5em]{.}\contentspage} % 页码和引导符

% 超链接颜色设置
\hypersetup{
    colorlinks=true,       % 激活颜色链接（去掉框，改为字变色）
    linkcolor=black,       % 内部链接（如目录、公式引用）为黑色
    citecolor=black,       % 文献引用为黑色
    urlcolor=blue,         % 网页链接为蓝色
    bookmarksnumbered=true % 书签带章节编号
}


%% --- 设置公式与正文的间距 ---
\AtBeginDocument{
    % 1. 公式上方默认间距
    \setlength{\abovedisplayskip}{2.5pt}

    % 2. 公式下方默认间距
    \setlength{\belowdisplayskip}{2.5pt}

    % 3. 公式上方“短行”间距 
    % (当公式前一行的文字很短，没有碰到公式时，用这个较小的间距)
    \setlength{\abovedisplayshortskip}{0pt}

    % 4. 公式下方“短行”间距
    \setlength{\belowdisplayshortskip}{3pt}
}
% 让公式编号跟随章节 (效果：1.1, 1.2, 2.1)
\numberwithin{equation}{section}

% --- 设置表格全局格式 (表标题 + 表内容) ---

% 1. 设置表标题格式：五号、中文宋体、英文 Times New Roman
% \DeclareCaptionFont 定义一个新字体命令 \tablecapfont
% 逻辑：\zihao{5}设置字号 -> \songti设置中文为宋体 -> \rmfamily强制英文回退到衬线体(即TNR)
\DeclareCaptionFont{tablecapfont}{\zihao{5}\songti\rmfamily}
% 表格编号跟随章节 (效果：表 1.1, 表 1.2)
\numberwithin{table}{section}
% 应用到所有表格 (table) 的标题
\captionsetup[table]{
    font=tablecapfont, % 使用上面定义的字体
    labelsep=space,     % 序号和标题之间空一格 (1.1 标题)
    skip=1pt           % 标题和表格之间的垂直间距
}
\setlength{\heavyrulewidth}{0.75pt} % 顶线和底线
\setlength{\lightrulewidth}{0.5pt}  % 中间线
% 2. 设置表格内容格式：五号、中文宋体、英文 Times New Roman
% 使用 etoolbox 的 \AtBeginEnvironment 钩子
% 每当遇到 \begin{tabular} 时，自动执行后面的字体命令
\AtBeginEnvironment{tabular}{\zihao{5}\songti\rmfamily}
\AtBeginEnvironment{tabularx}{\zihao{5}\songti\rmfamily}


%% --- 设置图片标题格式 ---
% ==================================================
%  图片配置 (复制到 myclass.cls 末尾)
% ==================================================

% 1. 让图片编号跟随章节 (效果：图 1.1, 图 1.2, 图 2.1)
\numberwithin{figure}{section}

% 2. 设置图片标题格式
% 复用之前定义的 tablecapfont (五号、宋体、英文TNR)
% 如果你之前没定义过 tablecapfont，请先补充定义：
% \DeclareCaptionFont{tablecapfont}{\zihao{5}\songti\rmfamily}

\captionsetup[figure]{
    font=tablecapfont, % 使用五号字体
    labelsep=space,    % 序号后空一格 (图1.1 标题)
    skip=0pt,          % 图片和标题之间的距离
    position=bottom    % 标题通常在下方
}
