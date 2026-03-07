require "test_helper"

class EndpointsControllerTest < ActionDispatch::IntegrationTest
  test "should get index" do
    get endpoints_index_url
    assert_response :success
  end
end
