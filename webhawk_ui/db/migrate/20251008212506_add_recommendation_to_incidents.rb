class AddRecommendationToIncidents < ActiveRecord::Migration[8.0]
  def change
    add_column :incidents, :recommendation, :text
  end
end
