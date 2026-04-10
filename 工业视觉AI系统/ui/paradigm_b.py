"""Paradigm B UI rendering."""

from __future__ import annotations

import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.padim import build_padim_stats, compute_dist_map
from core.cv_utils import overlay_single_mask_on_image_rgb
from ui.components import UIComponents, LoadingStates  # ✅ 导入组件库

from ui.constants import WEB_MODEL_PATH, IMG_SIZE, FEAT_DIM
from ui.adapters import run_sam3_instance_segmentation, process_single_image


def render(*, device: str, sam_proc, sam_model, sam_dtype, resnet) -> None:
    st.markdown("### 范式 B：离线异常检测 (PaDiM + SAM-3 Purify)")

    with st.sidebar:
        st.markdown("## ⚙️ 参数设置")

        # === 基础配置 ===
        with st.expander(" 基础配置", expanded=True):
            target_prompt = st.text_input(
                "目标物体名称",
                value="Transistor",
                placeholder="Transistor / Screw / PCB",
                help="定义要检测的目标物体类型"
            )

            sam_thr = st.slider(
                "SAM-3 分割阈值",
                0.0, 1.0, 0.25,
                help="🎚️ 控制目标定位的敏感度，值越高越严格"
            )

        # === ROI 提取策略 ===
        with st.expander("📦 ROI 提取策略", expanded=True):
            roi_mode = st.selectbox(
                "ROI 生成模式",
                options=["bbox", "mask"],
                index=0,
                format_func=lambda x: {
                    "bbox": "🔲 BBox（含背景）",
                    "mask": "✂️ Mask（纯物体）"
                }[x]
            )

            # 根据选择显示不同的说明
            if roi_mode == "bbox":
                st.caption("💡 保留背景信息，适合检测空间/装配异常")
            else:
                st.caption("💡 仅保留物体本身，适合检测表面缺陷")

            context_pad = st.slider(
                "上下文扩充比例",
                0.0, 0.6, 0.2,
                help="扩大 ROI 范围以包含更多上下文"
            )

        # === 异常判定参数 ===
        with st.expander(" 异常判定参数", expanded=True):
            decision_thr = st.slider(
                "异常判定阈值",
                10.0, 100.0, 25.0,
                help="超过此值判定为异常"
            )

            use_auto_thr = st.toggle(
                "使用自动阈值",
                value=False,
                help="启用后使用训练集统计的自动阈值"
            )

            if use_auto_thr:
                auto_thr_percentile = st.slider(
                    "阈值百分位",
                    90, 100, 99,
                    help="基于训练集分数分布的百分位数"
                )
            else:
                auto_thr_percentile = 99

        # === 可视化参数 ===
        with st.expander("🎨 可视化参数", expanded=False):
            heat_alpha = st.slider(
                "热力图透明度",
                0.0, 1.0, 0.4,
                help="控制热力图叠加透明度"
            )

            blur_ksize = st.slider(
                "热力图平滑度",
                1, 41, 17,
                step=2,
                help="值越大越平滑（需为奇数）"
            )

        st.markdown("---")
        st.info(f"🖥️ 当前设备: **{device.upper()}**")

    tab_train, tab_test = st.tabs(["📚 训练正常样本", "🔍 检测异常样本"])

    with tab_train:
        st.markdown("## 📚 训练阶段")
        st.info("📝 **训练流程**: 上传 5-10 张正常样本 → 系统自动提取特征 → 建立统计模型 → 生成自动阈值")

        st.markdown("---")
        st.markdown("### 📤 批量上传正常样本")

        train_files = st.file_uploader(
            "选择正常样本图片（支持多选）",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg'],
            key="b_train",
            help="建议上传 5-10 张正常样本以获得更好的统计模型"
        )

        if train_files:
            st.success(f"✅ 已选择 **{len(train_files)}** 张图片")
            with st.expander("📋 查看文件列表", expanded=False):
                for i, f in enumerate(train_files[:10], 1):
                    st.caption(f"{i}. {f.name}")
                if len(train_files) > 10:
                    st.caption(f"... 以及其他 {len(train_files) - 10} 个文件")

        st.markdown("---")

        if st.button("🚀 开始训练", key="b_train_btn", type="primary", use_container_width=True, disabled=not bool(st.session_state.get("models_ready"))):
            if not train_files:
                st.error("❌ 请先上传图片！")
            else:
                st.markdown("---")
                st.markdown("### ⏳ 训练进度")

                feats_list = [[] for _ in range(256)]
                bar = st.progress(0)
                status_text = st.empty()

                train_feat_cache = []
                train_name_cache = []

                valid_count = 0
                for i, f in enumerate(train_files):
                    status_text.text(f"📊 正在处理: {f.name} ({i+1}/{len(train_files)})")

                    img = Image.open(f).convert("RGB")
                    feat, _ = process_single_image(
                        image_pil=img,
                        sam_proc=sam_proc,
                        sam_model=sam_model,
                        sam_dtype=sam_dtype,
                        resnet=resnet,
                        prompt=target_prompt,
                        threshold=sam_thr,
                        context_pad=context_pad,
                        roi_mode=roi_mode,
                        device=device,
                    )

                    if feat is not None:
                        valid_count += 1
                        train_feat_cache.append(feat)
                        train_name_cache.append(getattr(f, "name", f"sample_{i:03d}"))
                        feat_flat = feat.reshape(FEAT_DIM, -1)
                        for p in range(256):
                            feats_list[p].append(feat_flat[:, p])

                    bar.progress((i + 1) / len(train_files))

                status_text.text("正在计算统计模型...")

                if valid_count < 2:
                    bar.empty()
                    status_text.empty()
                    st.error("有效样本过少（至少需要 2 张能成功定位目标的图）。请检查 Prompt 或图片。")
                else:
                    means, inv_covs = build_padim_stats(feats_list, feat_dim=int(FEAT_DIM))

                    train_scores = []
                    for feat in train_feat_cache:
                        dist_map = compute_dist_map(feat, means, inv_covs)
                        dist_map_16 = dist_map.reshape(16, 16)
                        amap_train = cv2.resize(dist_map_16, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                        if blur_ksize >= 3:
                            amap_train = cv2.GaussianBlur(amap_train, (blur_ksize, blur_ksize), 0)
                        train_scores.append(float(np.max(amap_train)))

                    try:
                        auto_thr = float(np.percentile(np.array(train_scores, dtype=np.float32), auto_thr_percentile))
                    except Exception:
                        auto_thr = 25.0

                    np.savez(
                        WEB_MODEL_PATH,
                        means=np.array(means),
                        inv_covs=np.array(inv_covs),
                        feat_dim=int(FEAT_DIM),
                        auto_thr=auto_thr,
                        auto_thr_percentile=float(auto_thr_percentile),
                        train_scores=np.array(train_scores, dtype=np.float32),
                    )

                    bar.empty()
                    status_text.empty()

                    # 使用组件库的成功提示
                    LoadingStates.success_toast(
                        "🎉 模型训练完成！",
                        details=f"有效样本 {valid_count}/{len(train_files)}，已保存至 `{WEB_MODEL_PATH}`\n自动阈值 auto_thr(P{auto_thr_percentile}) = {auto_thr:.2f}"
                    )

                    st.markdown("---")
                    st.markdown("### 📊 训练结果统计")

                    with st.expander("📈 查看训练集分数分布", expanded=True):
                        st.caption(
                            "这里的 score 是每张正常样本在 PaDiM 距离图上的最大值（即 max heatmap，通常可视作‘最大马氏距离/异常强度’）。"
                        )

                        scores_np = np.array(train_scores, dtype=np.float32)
                        plot_done = False
                        if scores_np.size > 0:
                            try:
                                import matplotlib.pyplot as plt  # type: ignore

                                fig = plt.figure(figsize=(7.2, 4.2), dpi=120)
                                ax = fig.add_subplot(111)
                                ax.hist(scores_np, bins=min(30, max(8, int(scores_np.size))), color="#4C78A8", alpha=0.85, label="Normal Samples")
                                ax.axvline(float(auto_thr), color="red", linestyle="--", linewidth=2.0, label=f"Threshold: {float(auto_thr):.2f}")
                                ax.set_title("Training Set Score Distribution")
                                ax.set_xlabel("Score")
                                ax.set_ylabel("Count")
                                ax.grid(True, alpha=0.25)
                                ax.legend(loc="upper right")
                                fig.tight_layout()

                                out_path = os.path.join("paradigm_b", "train_distribution.png")
                                fig.savefig(out_path)
                                plt.close(fig)

                                st.image(out_path, use_container_width=True)
                                plot_done = True
                            except Exception:
                                plot_done = False

                        if not plot_done:
                            if scores_np.size > 0:
                                bins = int(min(20, max(5, scores_np.size)))
                                hist, edges = np.histogram(scores_np, bins=bins)
                                centers = (edges[:-1] + edges[1:]) / 2

                                try:
                                    import pandas as pd  # type: ignore

                                    hist_df = pd.DataFrame({
                                        "score_bin_center": centers,
                                        "count": hist,
                                    })
                                    st.write("训练分数直方图（count vs score）")
                                    st.bar_chart(hist_df.set_index("score_bin_center"))
                                except Exception:
                                    st.write("训练分数直方图（count vs score）")
                                    st.bar_chart(hist)

                                st.caption(
                                    f"阈值说明：auto_thr 取训练分数的 P{int(auto_thr_percentile)} 分位数 = {auto_thr:.3f}。"
                                )
                            else:
                                st.info("暂无训练分数可绘制。")

                        st.markdown("---")
                        st.markdown("#### 📋 训练样本分数 Top-20")

                        n_pairs = min(len(train_name_cache), len(train_scores))
                        rows = [
                            {"image": str(train_name_cache[i]), "score": float(train_scores[i])}
                            for i in range(n_pairs)
                        ]

                        rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
                        top_rows = rows_sorted[:20]

                        try:
                            import pandas as pd  # type: ignore

                            df = pd.DataFrame(rows_sorted)
                            st.dataframe(
                                df.head(20),
                                use_container_width=True,
                                hide_index=True,
                            )
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                        except Exception:
                            st.dataframe(top_rows, use_container_width=True)
                            csv_lines = ["image,score"] + [f"{r['image']},{r['score']}" for r in rows_sorted]
                            csv_bytes = ("\n".join(csv_lines)).encode("utf-8")

                        st.download_button(
                            label="下载训练分数 CSV",
                            data=csv_bytes,
                            file_name="train_scores.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        # 使用组件库的统计面板
                        UIComponents.statistics_panel({
                            "min": f"{float(np.min(train_scores)):.3f}",
                            "mean": f"{float(np.mean(train_scores)):.3f}",
                            "max": f"{float(np.max(train_scores)):.3f}",
                            f"auto_thr (P{int(auto_thr_percentile)})": f"{auto_thr:.3f}"
                        }, cols=4)

                        st.info(
                            "提示：这里的‘训练分数’不是把 256×256 的所有像素做平均，而是取热力图最大值（max）。"
                            "这样对局部缺陷最敏感。若你想更抗噪，可扩展为 top‑p 平均等聚合方式。"
                        )

    with tab_test:
        st.markdown("## 🔍 检测阶段")
        st.info("📝 **检测流程**: 上传测试图片 → SAM-3 定位目标 → PaDiM 异常分析 → 生成热力图 → 判定结果")

        st.markdown("---")
        st.markdown("### 📤 上传测试图片")

        test_file = st.file_uploader(
            "选择待检测图片",
            type=['png', 'jpg', 'jpeg'],
            key="b_test",
            help="上传需要检测的图片"
        )

        if test_file:
            test_img = Image.open(test_file).convert("RGB")
            st.session_state["b_raw_rgb"] = np.array(test_img)

            st.success(f"✅ 已加载: {test_file.name}")

            st.markdown("---")
            st.markdown("## 📊 分析结果")

            b_left, b_right = st.columns(2, gap="large")
            with b_left:
                st.markdown("### 🖼️ 原始图像")
                st.image(test_img, use_container_width=True)

            st.markdown("---")
            st.markdown("### ⚙️ 检测控制")

            auto_run_b = st.toggle(
                "上传后自动检测",
                value=False,
                help="默认关闭：避免上传/刷新时自动耗时推理。开启后：更换新图会自动执行一次检测。",
                key="b_auto_run"
            )

            rerun_needed = st.session_state.get("b_last_uploaded_name") != test_file.name

            run_b = st.button("🔍 执行检测", type="primary", key="b_run", use_container_width=True, disabled=not bool(st.session_state.get("models_ready")))

            should_run = bool(run_b) or (bool(auto_run_b) and bool(rerun_needed))

            if should_run:
                if not os.path.exists(WEB_MODEL_PATH):
                    st.error("请先在 Tab 1 训练模型！")
                else:
                    # 使用组件库的加载状态
                    with LoadingStates.spinner("正在进行 SAM-3 定位与 PaDiM 分析..."):
                        sam_results, _ = run_sam3_instance_segmentation(
                            image_pil=test_img,
                            sam_proc=sam_proc,
                            sam_model=sam_model,
                            sam_dtype=sam_dtype,
                            prompt=str(target_prompt),
                            threshold=float(sam_thr),
                            device=device,
                        )

                        feat, roi = process_single_image(
                            image_pil=test_img,
                            sam_proc=sam_proc,
                            sam_model=sam_model,
                            sam_dtype=sam_dtype,
                            resnet=resnet,
                            prompt=target_prompt,
                            threshold=sam_thr,
                            context_pad=context_pad,
                            roi_mode=roi_mode,
                            device=device,
                        )

                    st.session_state["b_last_feat"] = feat
                    st.session_state["b_last_roi"] = roi
                    st.session_state["b_last_uploaded_name"] = test_file.name
                    st.session_state["b_last_sam_results"] = sam_results

            feat = st.session_state.get("b_last_feat")
            roi = st.session_state.get("b_last_roi")
            sam_results = st.session_state.get("b_last_sam_results")

            with b_right:
                st.markdown("###  检测结果")

                if roi is None or feat is None:
                    st.warning("⚠️ 暂无可用结果。请点击「执行检测」按钮，或检查 Prompt/阈值设置。")
                else:
                    model_data = np.load(WEB_MODEL_PATH)

                    saved_dim = int(model_data['feat_dim']) if 'feat_dim' in model_data else None
                    if saved_dim is not None and saved_dim != int(FEAT_DIM):
                        st.error(
                            f"模型特征维度不匹配：文件={saved_dim}，当前={FEAT_DIM}。请重新训练模型。"
                        )
                        st.stop()

                    means, inv_covs = model_data['means'], model_data['inv_covs']

                    effective_decision_thr = float(decision_thr)
                    if use_auto_thr and 'auto_thr' in model_data:
                        effective_decision_thr = float(model_data['auto_thr'])
                        st.info(f" 当前判定阈值: **auto_thr = {effective_decision_thr:.2f}** (来自训练集)")
                    else:
                        st.info(f" 当前判定阈值: **decision_thr = {effective_decision_thr:.2f}** (手动设置)")

                    st.markdown("---")
                    st.markdown("#### 📍 SAM-3 定位与 ROI 预览")

                    if sam_results and len(sam_results.get("masks", [])) > 0:
                        raw_rgb = st.session_state.get("b_raw_rgb")
                        scores = sam_results["scores"].float().cpu().numpy()
                        best_idx = int(np.argmax(scores))
                        best_mask = sam_results["masks"][best_idx].cpu().numpy() > 0.5

                        vis_overlay = overlay_single_mask_on_image_rgb(raw_rgb, best_mask, alpha=0.45, color=(0, 255, 0))

                        v1, v2 = st.columns(2)
                        with v1:
                            st.image(
                                vis_overlay,
                                use_container_width=True,
                                caption=f"✅ SAM-3 定位掩码 (置信度={float(scores[best_idx]):.3f})",
                            )
                        with v2:
                            st.image(
                                roi,
                                use_container_width=True,
                                caption=f"📦 提取的 ROI (256×256) | 模式={roi_mode} | 扩充={context_pad:.2f}",
                            )

                    else:
                        st.warning("⚠️ SAM-3 未返回有效 mask，无法展示 overlay/ROI。请尝试调整 Prompt 或 SAM-3 阈值。")

                    st.markdown("---")
                    st.markdown("#### 🔥 异常热力图分析")

                    dist_map = compute_dist_map(feat, means, inv_covs)
                    dist_map = dist_map.reshape(16, 16)
                    amap = cv2.resize(dist_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
                    if blur_ksize >= 3:
                        amap = cv2.GaussianBlur(amap, (blur_ksize, blur_ksize), 0)
                    score = float(np.max(amap))

                    show_c1, show_c2 = st.columns(2)
                    with show_c1:
                        st.image(roi, caption="📦 ROI 输入 (256×256)", use_container_width=True)
                    with show_c2:
                        norm_map = (amap - float(amap.min())) / (float(amap.max() - amap.min()) + 1e-8)
                        heatmap = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(
                            cv2.cvtColor(roi, cv2.COLOR_RGB2BGR), 1 - heat_alpha, heatmap, heat_alpha, 0
                        )
                        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="🔥 异常热力图叠加", use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### 🎯 判定结果")

                    if score > effective_decision_thr:
                        st.error(f"🔴 **发现缺陷** | 异常分数: **{score:.2f}** > 阈值: {effective_decision_thr:.2f}")
                    else:
                        st.success(f"✅ **正常样本** | 异常分数: **{score:.2f}** ≤ 阈值: {effective_decision_thr:.2f}")
"""Streamlit UI layer modules.

This package hosts UI rendering code (pages/sections) and session-state initialization.
Core algorithms and model code live in `core/`.
"""

