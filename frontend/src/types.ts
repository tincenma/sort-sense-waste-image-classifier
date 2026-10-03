export type Category = 'cardboard' | 'glass' | 'metal' | 'paper' | 'plastic' | 'trash'
export type ModelId = 'mobilenetv2' | 'baseline_cnn'

export interface ModelInfo {
  id: ModelId
  name: string
  description: string
  test_accuracy: number
  test_macro_f1: number
}

export interface Example {
  id: Category
  label: Category
  image_url: string
}

export interface AppInfo {
  categories: Category[]
  models: ModelInfo[]
  examples: Example[]
  max_file_mb: number
}

export interface Prediction {
  model_id: ModelId
  predicted_category: Category
  model_score: number
  category_scores: Record<Category, number>
  inference_ms: number
}

export interface PredictionResponse {
  filename: string
  width: number
  height: number
  total_ms: number
  predictions: Prediction[]
}
