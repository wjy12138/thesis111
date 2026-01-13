% myclass.cls
% 版本：v1.0
% 描述：基础页面布局配置

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{myclass}[2024/01/01 v1.0 My Custom LaTeX Class]

% 1. 加载基础类 (使用 article，纸张默认为 A4)
\LoadClass[a4paper, zihao=-4]{ctexart}

% 2. 加载页面布局宏包
\RequirePackage{geometry}  % 页面布局
\RequirePackage{graphicx}  % 图片支持
\RequirePackage{setspace}  % 行距控制
\RequirePackage{fontspec}  % 字体设置
\RequirePackage{fancyhdr} % 标题格式控制
\RequirePackage{hyperref} % 超链接支持
\RequirePackage{titletoc}  % 用于自定义目录格式
\RequirePackage{titlesec}  % 用于自定义章节标题格式
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
\fancypagestyle{plain}{
    \fancyhf{}
    \fancyhead[L]{\songti\zihao{-5} 重庆大学本科学生毕业论文（设计）}
    \fancyhead[R]{\songti\zihao{-5} 正文}
    \renewcommand{\headrulewidth}{0.4pt}
    \renewcommand{\footrulewidth}{0pt}
    \fancyfoot[C]{\thepage}
}

% 7. 设置全局页面样式
\pagestyle{plain}

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
        { \heiti \zihao{2} XXX计算方法的研究 \par}
        
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
                
                % 【关键修正2】盒子宽度改为 6em (适配 "助理指导教师" 6个字)
                % [s] 表示分散对齐
                \newcommand{\fieldlabel}[1]{\makebox[4em][s]{##1}} 

                % 逐行输出
                \noindent \fieldlabel{学生}：王建华 \par
                \noindent \fieldlabel{学号}：20XXXXXX \par
                \noindent \fieldlabel{指导教师}：杨X \par
                \noindent \fieldlabel{助理指导}教师：李X \par
                \noindent \fieldlabel{专业}：XXXXXX \par
            \end{flushleft}
        }

        % --- 空两行 (三号) ---
        % 由于上面用了 flushleft，这里需要恢复居中，或者直接用 center 环境
        % 注意：为了保证垂直间距准确，我们退出 flushleft 后加间距
        {\zihao{3}  \setlength{\baselineskip}{31pt} \vspace{2\baselineskip}}

        % --- 5. 学院名称 ---
        % 小二，居中
        {\centering \heiti \zihao{-2} 重庆大学XXXXXX学院 \par}

        % --- 6. 日期 ---
        % 黑体三号，段前12磅，居中
        \vspace{21pt}
        {\centering \heiti \zihao{3} 20XX年6月 \par}
        
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
        { \rmfamily \bfseries \zihao{2} Study on Calculation method of XXX \par}
        
        % --- 空三行 (五号) ---
        { \zihao{5} \setlength{\baselineskip}{17pt} \vspace{3\baselineskip}}
        
        % --- 3. 图片 / Logo ---
        \includegraphics[width=4.27cm, height=4.19cm]{figure/logo.png}

        % --- 空两行 (三号) ---
        { \zihao{3} \setlength{\baselineskip}{21.5pt} \vspace{2\baselineskip}}
        
        % --- 学生个人信息 ---
        { \rmfamily \bfseries \zihao{3} By \vspace{16pt}}

        { \rmfamily \bfseries \zihao{3} WANG Jianhua }

        { \zihao{3} \setlength{\baselineskip}{32pt} \vspace{1\baselineskip} }

        { \rmfamily \bfseries \zihao{3} Supervised by \vspace{17pt}}

        { \rmfamily \bfseries \zihao{3} Prof. YANG XX \vspace{16pt} }

        { \rmfamily \bfseries \zihao{3} and \vspace{17pt}}

        { \rmfamily \bfseries \zihao{3} Prof. LI XX }

        { \zihao{3} \setlength{\baselineskip}{33pt} \vspace{1\baselineskip} }

        % 专业    
        { \rmfamily \bfseries \zihao{-2} XXXXXX \vspace{17pt}}

        % 学院    
        { \rmfamily \bfseries \zihao{-2} XXXXXX }

        % --- 空两行 (三号) ---
        {\zihao{3}  \setlength{\baselineskip}{12pt} \vspace{1\baselineskip}}

        % --- 5. 学院名称 / School Name ---
        % 【修改】字体改为 Times New Roman 加粗
        {\centering \rmfamily \bfseries \zihao{-2} Chongqing University \par}

        % --- 6. 日期 / Date ---
        \vspace{10pt}
        % 【修改】字体改为 Times New Roman 加粗
        { \centering \rmfamily \bfseries \zihao{3} June, 20XX \par}
        
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
    \setlength{\leftskip}{2.2em} % ← 这里调整整行的左缩进
    \noindent
    {\heiti \zihao{-4} 关键词：}
    {\songti \zihao{-4} #1}
    
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
    \setlength{\leftskip}{2.2em} % 与中文保持相同缩进
    \noindent
    {\rmfamily \bfseries \zihao{-4} Keywords: } % 加粗
    {\rmfamily \zihao{-4} #1} % 不加粗
    
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
  {\vspace*{-5.2pt}\centering\heiti\fontspec{SimHei}\zihao{3}}{\thesection}{0.6em}{}  %编号和标题之间的距离为0.5em，顶部距离，强制英文黑体
% % --- 2. 【新增】修改一级标题间距 ---
% % \titlespacing{命令}{左缩进}{段前间距}{段后间距}
% \titlespacing{\section}{0.0001pt}{*36pt}{18pt}

\titleformat{\subsection}
  {\heiti\zihao{-3}\fontspec{SimHei}}{\thesubsection}{0.6em}{}

\titleformat{\subsubsection}
  {\heiti\zihao{-3}\fontspec{SimHei}}{\thesubsubsection}{0.6em}{}

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
        \heiti  \zihao{3} 目\quad 录
    \end{center}
    
    \vspace{6.5pt} % 与摘要保持相同的间距
    
    % 目录正文：固定行距20磅
    \setlength{\baselineskip}{20pt}
    \selectfont
    % 开始生成目录
    \@starttoc{toc}
    
    % 确保目录后正确分页
    \clearpage
}

% 设置目录条目格式
% 一级标题（section）：黑体4号
\titlecontents{section}
  [2em] % 左缩进
  {\addvspace{0pt}\heiti\zihao{4}} % 格式
  {\contentslabel{1.5em}} % 带编号的标签
  {\hspace*{-1.5em}} % 无编号的标签
  {\titlerule*[0.4em]{.}\contentspage} % 页码和引导符
  
% 二级标题（subsection）：黑体5号
\titlecontents{subsection}
  [2.7em] % 左缩进（比一级标题缩进2em）
  {\addvspace{0pt}\heiti\zihao{5}} % 格式
  {\contentslabel{2em}} % 带编号的标签
  {\hspace*{-2.7em}} % 无编号的标签
  {\titlerule*[0.6em]{.}\contentspage} % 页码和引导符

% 三级标题（subsubsection）：宋体5号
\titlecontents{subsubsection}
  [4em] % 左缩进（比二级标题再缩进2em）
  {\addvspace{0pt}\songti\zihao{5}} % 格式
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