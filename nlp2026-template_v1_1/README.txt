# nlp2026-template_v1_1.zip 内のファイル

Latex版
 nlp2026.cls: latex用の文書クラス
 nlp2026-sample.tex: 原稿執筆インストラクション，および，latex用のサンプル原稿
 nlp2026-sample.pdf: nlp2026-sample.texをPDF化したもの
 j_yourrefs.bib: サンプルbibファイル

Word版
 nlp2026-wordsample.doc: Word用のサンプル原稿 (可能な限り.docxの利用を推奨）
 nlp2026-wordsample.docx: Word用のサンプル原稿 
 nlp2026-wordsample.pdf: nlp2026-wordsample.docxをPDF化したもの

参考ファイル
 LICENSE: nlp2026.cls のライセンス条項
 latexmkrc: Overleafの日本語化用設定ファイルのサンプル
 README-latex.md: nlp2026.cls に関する説明文書
 README.txt: このファイル


更新履歴

* v2026.1_1 2025.12.11:
 - README.txt, README_latex.md の修正
  
* v2026.1_0 2025.10.28: 2026年度初期バージョン
 - Word版参考文献の変更

* v2025.1_1 2024.12.22:
 - LuaLaTeX でのフォントの問題に対処

* v2025.1_0 2024.10.29: 2025年度初期バージョン

* v2024.1_0 2023.11.8: 2024年度初期バージョン

* v2023.1_0 2022.11.7: 2023年度初期バージョン

* v2022.1_1 2021.12.25: 
 - 英語対応版(English オプション)利用時の「概要」の表記を「Abstract」に変更

* v2022.1_0 2021.12.7: 2022年度版初期バージョン
 - 概要（新要素）に関する説明の追記
 - 多書体化のために jlreq-deluxe を導入
 - 謝辞を参考文献のページに含めてよい（本文とみなさない）ことの仕様変更に関する説明を追記

* v2021.3 2020.12.22:
 - latex版の参考文献のフォントサイズ指定方法の変更
 - natbibの廃止(挙動の制御が難しいため)
 - 参考文献のフォントサイズの設定コマンドを追加
   - 例：\renewcommand{\bibfont}{\footnotesize} 


* v2021.2 2020.12.18: 
 - english オプションの追加


* v2021.1 2020.12.3: 
 - 参考文献の例を \bibliographystyle{unsrt} => \bibliographystyle{junsrt} に変更
 - nlp2021-sample.tex 内の文言を一部修正（規定には影響なし）


* v20201.0 2020.12.1: 初期バージョン
