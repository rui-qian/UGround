"""
智能模板选择器 - 解决多目标分割模板选择中的语法和冗余问题

主要特性：
1. 语法智能：正确处理单复数和动词变位
2. 指令检测：避免重复添加分割指令
3. 多目标支持：智能处理多个分割目标
4. Token管理：支持SEG/REJ token的灵活配置
"""

import re
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Union

# 避免复杂导入，直接定义常量
DEFAULT_IMAGE_TOKEN = "<image>"


@dataclass
class TemplateContext:
    """模板上下文信息封装"""
    class_names: Union[str, List[str]]  # 类名，可以是单个字符串或列表
    masks: Optional[List[Any]] = None  # 掩码列表，用于验证有效性
    is_sentence: bool = False  # 是否为句子形式
    seg_token_num: int = 1  # SEG token数量
    existing_text: Optional[str] = None  # 已有的文本（用于检测指令冗余）
    use_separate_answers: bool = False  # 是否使用分离式回答
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.class_names, str):
            self.class_names = [self.class_names]


class SmartTemplateSelector:
    """智能模板选择器"""
    
    def __init__(self):
        """初始化模板选择器"""
        # 单复数不规则变化词典
        self.irregular_plurals = {
            'child': 'children',
            'person': 'people', 
            'man': 'men',
            'woman': 'women',
            'mouse': 'mice',
            'foot': 'feet',
            'tooth': 'teeth',
            'goose': 'geese'
        }
        
        # 逆向映射
        self.irregular_singulars = {v: k for k, v in self.irregular_plurals.items()}
        
        # 分割指令关键词（更精确的匹配）
        self.segmentation_keywords = [
            'segment', 'segmentation', 'mask', 'masking', 'identify', 'locate', 
            'detect', 'point out', 'highlight', 'mark', 'outline', 'trace', 
            'draw mask', 'create mask', 'generate mask', 'provide mask',
            'extract', 'isolate', 'separate', 'distinguish', 'pinpoint',
            'pixel-level', 'contour', 'boundary', 'region'
        ]
        
        # 模板定义
        self._init_templates()
    
    def _init_templates(self):
        """初始化所有模板"""
        # 分割指令关键词
        self.segmentation_keywords = [
            'segment', 'identify', 'locate', 'find', 'detect', 'point out', 
            'highlight', 'mark', 'outline', 'trace', 'show', 'draw', 'mask',
            'select', 'extract', 'isolate', 'separate', 'distinguish', 'pinpoint'
        ]
        
        # 短问题模板（大幅扩展多样性）
        self.short_question_templates = [
            # 基础形式
            DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Where {verb} {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Please identify {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Could you locate {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Can you find {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Please segment {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Show me {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Highlight {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Point out {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Mark {class_name} for me.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Detect {class_name} in this image.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Extract {class_name} from the image.",
            
            # 礼貌形式
            DEFAULT_IMAGE_TOKEN + "\n" + "Could you please segment {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Would you mind identifying {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "I'd like to see {class_name} segmented.",
            DEFAULT_IMAGE_TOKEN + "\n" + "May I ask you to find {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Please be so kind as to locate {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "If possible, could you highlight {class_name}?",
            
            # 描述性形式
            DEFAULT_IMAGE_TOKEN + "\n" + "I need the segmentation of {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "I'm looking for {class_name} in this image.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Help me identify where {class_name} {verb}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "I want to see the outline of {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Show me the boundaries of {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "I need to locate {class_name} precisely.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Can you isolate {class_name} for me?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Please separate {class_name} from the background.",
            
            # 任务导向形式
            DEFAULT_IMAGE_TOKEN + "\n" + "Segment {class_name} from this image.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Create a mask for {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Generate segmentation for {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Provide the mask of {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Output the segmentation of {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Draw the contour of {class_name}.",
            
            # 疑问形式
            DEFAULT_IMAGE_TOKEN + "\n" + "What parts of the image {verb} {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Which regions contain {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "How can I segment {class_name}?",
            DEFAULT_IMAGE_TOKEN + "\n" + "Where exactly {verb} {class_name} located?",
            DEFAULT_IMAGE_TOKEN + "\n" + "What's the shape of {class_name}?",
            
            # 专业术语形式
            DEFAULT_IMAGE_TOKEN + "\n" + "Perform semantic segmentation on {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Execute instance segmentation for {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Apply segmentation algorithm to {class_name}.",
            DEFAULT_IMAGE_TOKEN + "\n" + "Compute pixel-level classification for {class_name}.",
        ]
        
        # 长问题模板（句子形式，大幅扩展）
        self.long_question_templates = [
            # 基础形式
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please provide segmentation.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you segment this?",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Show me the segmentation.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I need the mask.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Could you provide a segmentation map?",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Show me the segmentation result.",
            
            # 详细描述形式
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please segment the mentioned objects.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I'd like to see the segmentation of these items.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Can you create masks for what's described?",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please identify and segment accordingly.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Generate segmentation based on this description.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Provide pixel-level identification.",
            
            # 任务指向形式
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Execute segmentation task.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Perform semantic analysis.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Apply computer vision segmentation.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Create detailed object masks.",
            
            # 交互形式
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Help me with segmentation.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I need your assistance with masking.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Could you help identify the regions?",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please assist with object detection.",
            
            # 分析形式
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Analyze and segment.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Process this image request.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Interpret and provide masks.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Parse and segment accordingly.",
        ]
        
        # 长问题模板（无指令冗余）
        self.long_question_templates_simple = [
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent}",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Thank you.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please help.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I appreciate your assistance.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} That would be helpful.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Much appreciated.",
            DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Thanks in advance.",
        ]
        
        # 单目标回答模板（大幅扩展）
        self.single_answer_templates = [
            # 简洁形式
            "It is [SEG].",
            "Sure, it is [SEG].", 
            "Here it is: [SEG].",
            "Found it: [SEG].",
            "Located: [SEG].",
            "Identified: [SEG].",
            
            # 包含类名形式
            "{class_name} is [SEG].",
            "The {class_name} is [SEG].",
            "Here is the {class_name}: [SEG].",
            "The segmentation of {class_name} is [SEG].",
            "I found the {class_name}: [SEG].",
            "The {class_name} appears at [SEG].",
            "The location of {class_name} is [SEG].",
            "{class_name} can be found at [SEG].",
            "The {class_name} region is [SEG].",
            
            # 描述性形式
            "I can see the {class_name} at [SEG].",
            "The {class_name} is located at [SEG].",
            "The {class_name} boundary is [SEG].",
            "The outline of {class_name} is [SEG].",
            "The shape of {class_name} is [SEG].",
            "The contour of {class_name} is [SEG].",
            
            # 确认形式
            "Yes, I found {class_name}: [SEG].",
            "Absolutely, the {class_name} is [SEG].",
            "Certainly, {class_name} is at [SEG].",
            "Of course, here's the {class_name}: [SEG].",
            "Indeed, the {class_name} is [SEG].",
            
            # 专业形式
            "Segmentation result for {class_name}: [SEG].",
            "Mask output for {class_name}: [SEG].",
            "Pixel classification for {class_name}: [SEG].",
            "Object detection result: {class_name} at [SEG].",
            "Instance segmentation: {class_name} is [SEG].",
            
            # 完成形式
            "Done! The {class_name} is [SEG].",
            "Complete! {class_name} segmented as [SEG].",
            "Finished! The {class_name} mask is [SEG].",
            "Ready! {class_name} is identified as [SEG].",
        ]
        
        # 多目标回答模板（组合式，大幅扩展）
        self.multi_answer_templates = [
            # 基础形式
            "{class_names} are {seg_tokens}.",
            "Sure, {class_names} are {seg_tokens}.",
            "The {class_names} are {seg_tokens}.",
            "Here are the segmentation results: {class_names} are {seg_tokens}.",
            "The segmentation results of {class_names} are {seg_tokens}.",
            "I can identify {class_names} as {seg_tokens}.",
            
            # 详细形式
            "I found multiple objects: {class_names} are {seg_tokens}.",
            "The segmented regions are: {class_names} at {seg_tokens}.",
            "Multiple segments detected: {class_names} correspond to {seg_tokens}.",
            "Object identification complete: {class_names} are {seg_tokens}.",
            "Segmentation analysis: {class_names} are located at {seg_tokens}.",
            
            # 列举形式
            "The detected objects {class_names} have masks {seg_tokens}.",
            "I can segment the following: {class_names} as {seg_tokens}.",
            "The identified elements {class_names} appear at {seg_tokens}.",
            "Multiple items found: {class_names} with regions {seg_tokens}.",
            
            # 确认形式
            "Yes, I see {class_names} at {seg_tokens}.",
            "Confirmed: {class_names} are segmented as {seg_tokens}.",
            "Absolutely, the {class_names} are {seg_tokens}.",
            
            # 专业形式
            "Multi-object segmentation: {class_names} → {seg_tokens}.",
            "Instance detection results: {class_names} = {seg_tokens}.",
            "Semantic segmentation output: {class_names} at {seg_tokens}.",
        ]
        
        # 多目标回答模板（分离式）
        self.multi_answer_templates_separate = [
            "{answers}",
            "Sure, {answers}",
            "Here are the results: {answers}",
            "The segmentation results are: {answers}",
            "I found the following: {answers}",
            "Detection complete: {answers}",
            "Object analysis: {answers}",
            "Segmentation summary: {answers}",
            "Individual results: {answers}",
            "Detailed breakdown: {answers}",
        ]
        
        # 否定回答模板（扩展）
        self.negative_answer_templates = [
            "Sorry, I cannot find {class_name} in this image.",
            "No {class_name} detected in this image.",
            "I don't see any {class_name} here.",
            "The image does not contain {class_name}.",
            "Unable to locate {class_name} in this image.",
            "No {class_name} visible in the current image.",
            "{class_name} is not present in this scene.",
            "I cannot identify {class_name} in this picture.",
            "The requested {class_name} is not found.",
            "Search for {class_name} returned no results.",
            "No instances of {class_name} detected.",
            "The image lacks {class_name}.",
            "Unfortunately, {class_name} is absent from this image.",
            "I'm unable to segment {class_name} as it's not visible.",
        ]
        
        # 纠正回答模板（虚假前提，扩展）
        self.correction_answer_templates = [
            "There is no {class_name}, but I found {correct_name}.",
            "I cannot locate {class_name}, however {correct_name} is present.",
            "The {class_name} is not in this image, but {correct_name} is visible.",
            "Actually, I don't see {class_name}, but I can identify {correct_name}.",
            "No {class_name} here, though I notice {correct_name}.",
            "Instead of {class_name}, I found {correct_name}.",
            "The image shows {correct_name}, not {class_name}.",
            "I see {correct_name} rather than {class_name}.",
            "Correction needed: this is {correct_name}, not {class_name}.",
            "The object appears to be {correct_name}, not {class_name}.",
        ]
    
    def _detect_plurality(self, text: str) -> bool:
        """检测文本是否为复数形式"""
        text = text.strip().lower()
        words = text.split()
        
        if not words:
            return False
            
        # 检查不规则复数
        if any(word in self.irregular_plurals.values() for word in words):
            return True
            
        # 检查常规复数规则
        last_word = words[-1]
        if (last_word.endswith('s') and not last_word.endswith('ss') and 
            len(last_word) > 1 and last_word not in self.irregular_singulars):
            return True
            
        return False
    
    def _get_correct_verb(self, class_names: List[str]) -> str:
        """根据类名数量和形式选择正确的动词"""
        if len(class_names) == 1:
            # 单个目标，检查是否为复数形式
            if self._detect_plurality(class_names[0]):
                return "are"
            else:
                return "is"
        else:
            # 多个目标总是用复数
            return "are"
    
    def _format_class_names(self, class_names: List[str]) -> str:
        """格式化类名列表为自然语言"""
        if len(class_names) == 0:
            return ""
        elif len(class_names) == 1:
            return class_names[0]
        elif len(class_names) == 2:
            return f"{class_names[0]} and {class_names[1]}"
        else:
            return ", ".join(class_names[:-1]) + f" and {class_names[-1]}"
    
    def _has_segmentation_instruction(self, text: str) -> bool:
        """检测文本中是否已包含分割指令（更精确的匹配）"""
        text_lower = text.lower()
        
        # 直接包含明确的分割词汇
        explicit_keywords = [
            'segment', 'segmentation', 'mask', 'masking', 'outline', 'trace',
            'point out', 'highlight', 'draw mask', 'create mask', 'generate mask',
            'provide mask', 'extract', 'isolate', 'delineate', 'contour',
            'boundary', 'pixel-level', 'region'
        ]
        
        # 检查明确的分割指令
        for keyword in explicit_keywords:
            if keyword in text_lower:
                return True
        
        # 检查上下文相关的词汇组合
        context_patterns = [
            ('identify', 'region'),
            ('locate', 'area'),
            ('detect', 'object'),
            ('mark', 'area'),
            ('separate', 'background')
        ]
        
        for word1, word2 in context_patterns:
            if word1 in text_lower and word2 in text_lower:
                return True
        
        return False
    
    def _generate_seg_tokens(self, count: int, seg_token_num: int, 
                           valid_masks: Optional[List[bool]] = None) -> List[str]:
        """生成SEG/REJ token列表"""
        tokens = []
        
        for i in range(count):
            # 检查mask有效性
            is_valid = valid_masks[i] if valid_masks else True
            
            if seg_token_num == 1:
                base_token = "[SEG]" if is_valid else "[REJ]"
            else:
                base_token = f"[SEG{i}]" if is_valid else f"[REJ{i}]"
            
            tokens.append(base_token)
        
        return tokens
    
    def _format_seg_tokens(self, tokens: List[str]) -> str:
        """格式化SEG token列表为自然语言"""
        if len(tokens) == 1:
            return tokens[0]
        elif len(tokens) == 2:
            return f"{tokens[0]} and {tokens[1]}"
        else:
            return ", ".join(tokens[:-1]) + f" and {tokens[-1]}"
    
    def generate_question(self, context: TemplateContext) -> str:
        """生成语法正确的问题"""
        if context.is_sentence:
            # 句子形式，检查是否已有分割指令
            if (context.existing_text and 
                self._has_segmentation_instruction(context.existing_text)):
                # 已有指令，使用简单模板
                template = random.choice(self.long_question_templates_simple)
                return template.format(sent=context.existing_text)
            else:
                # 无指令，使用完整模板
                template = random.choice(self.long_question_templates)
                return template.format(sent=context.existing_text or "{sent}")
        else:
            # 单词/短语形式
            if not context.class_names:
                return DEFAULT_IMAGE_TOKEN + "\n" + "Please specify what to segment."
            
            class_names_str = self._format_class_names(context.class_names)
            verb = self._get_correct_verb(context.class_names)
            
            template = random.choice(self.short_question_templates)
            return template.format(class_name=class_names_str, verb=verb)
    
    def generate_answer(self, context: TemplateContext) -> str:
        """生成语法正确的回答"""
        # 处理空类名列表的特殊情况
        if not context.class_names:
            return "I cannot generate an answer without class names."
        
        # 确定有效的mask
        valid_masks = None
        if context.masks:
            valid_masks = [mask.sum() > 0 if hasattr(mask, 'sum') else bool(mask) 
                          for mask in context.masks]
        
        # 生成SEG tokens
        seg_tokens = self._generate_seg_tokens(
            len(context.class_names), 
            context.seg_token_num, 
            valid_masks
        )
        
        if len(context.class_names) == 1:
            # 单目标
            template = random.choice(self.single_answer_templates)
            class_name = context.class_names[0]
            seg_token = seg_tokens[0]
            
            # 如果模板中没有占位符，使用简单形式
            if "{class_name}" not in template:
                return template.replace("[SEG]", seg_token)
            else:
                return template.format(class_name=class_name).replace("[SEG]", seg_token)
        
        else:
            # 多目标
            if context.use_separate_answers:
                # 分离式回答
                answers = []
                for i, class_name in enumerate(context.class_names):
                    template = random.choice(self.single_answer_templates)
                    if "{class_name}" in template:
                        answer = template.format(class_name=class_name)
                    else:
                        answer = template
                    answer = answer.replace("[SEG]", seg_tokens[i])
                    answers.append(answer.rstrip('.'))
                
                # 直接连接答案，而不是通过_format_class_names处理
                if len(answers) == 1:
                    combined_answers = answers[0]
                elif len(answers) == 2:
                    combined_answers = f"{answers[0]} and {answers[1]}"
                else:
                    combined_answers = ", ".join(answers[:-1]) + f", and {answers[-1]}"
                
                template = random.choice(self.multi_answer_templates_separate)
                return template.format(answers=combined_answers)
            
            else:
                # 组合式回答
                class_names_str = self._format_class_names(context.class_names)
                seg_tokens_str = self._format_seg_tokens(seg_tokens)
                
                template = random.choice(self.multi_answer_templates)
                return template.format(
                    class_names=class_names_str,
                    seg_tokens=seg_tokens_str
                )
    
    def generate_negative_answer(self, class_name: str) -> str:
        """生成否定回答"""
        template = random.choice(self.negative_answer_templates)
        return template.format(class_name=class_name)
    
    def generate_correction_answer(self, class_name: str, correct_name: str) -> str:
        """生成纠正回答（用于虚假前提情况）"""
        template = random.choice(self.correction_answer_templates)
        return template.format(class_name=class_name, correct_name=correct_name)
    
    def process_batch(self, contexts: List[TemplateContext]) -> List[Tuple[str, str]]:
        """批量处理多个模板上下文"""
        results = []
        for context in contexts:
            question = self.generate_question(context)
            answer = self.generate_answer(context)
            results.append((question, answer))
        return results


# 便利函数
def create_template_context(class_names, masks=None, is_sentence=False, 
                          seg_token_num=1, existing_text=None, 
                          use_separate_answers=False) -> TemplateContext:
    """创建模板上下文的便利函数"""
    return TemplateContext(
        class_names=class_names,
        masks=masks,
        is_sentence=is_sentence,
        seg_token_num=seg_token_num,
        existing_text=existing_text,
        use_separate_answers=use_separate_answers
    )


# 全局实例
smart_selector = SmartTemplateSelector() 