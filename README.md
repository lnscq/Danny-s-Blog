# Danny'Blog

一个基于 Astro 的个人网站/博客，记录日常、文章与项目。

## 技术栈

- Astro 5（静态站点与内容渲染）
- TypeScript
- Tailwind CSS + daisyUI（样式与主题）
- MDX/Markdown（内容写作）
- Pagefind（站内搜索索引）
- RSS/Sitemap（订阅与 SEO）
- Expressive Code / Shiki 主题（代码高亮）

## 功能概览

- 深浅色主题切换
- 文章分类与标签
- 站内搜索
- RSS 订阅与站点地图
- 代码高亮与数学公式渲染
- 响应式布局

## 目录结构（常用）

- `frosti.config.yaml`：站点基本信息与菜单配置
- `src/pages/`：页面与路由
- `src/content/`：文章内容（如有）
- `public/`：静态资源

## 开发与构建

```sh
pnpm install
pnpm run dev
```

首次运行或内容更新后，建议先生成搜索索引：

```sh
pnpm run search:index
```

构建与预览：

```sh
pnpm run build
pnpm run preview
```

## 配置说明

站点配置集中在 `frosti.config.yaml`，可修改标题、描述、菜单、头像与社交链接等。
