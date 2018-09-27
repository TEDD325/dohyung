/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller.integration;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.Action;
import org.edgexfoundry.controller.impl.CallbackExecutor;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresSpring.class, RequiresWeb.class})
public class CallbackExecutorTest {

  @Autowired
  private CallbackExecutor callback;

  private static int TEST_PORT = 9099;
  private static String TEST_PATH = "foo";

  @Test
  public void testCallbackWithAsset() throws IOException {
    HttpServer server = HttpServer.create(new InetSocketAddress(TEST_PORT), 0);
    server.createContext("/" + TEST_PATH, new TestHandler());
    server.setExecutor(null);
    server.start();
    Addressable addr = AddressableData.newTestInstance();
    addr.setPath(TEST_PATH);
    addr.setAddress("localhost");
    addr.setPort(TEST_PORT);
    Asset asset = ServiceData.newTestInstance();
    asset.setAddressable(addr);
    callback.callback(asset, "123", Action.POST, ActionType.DEVICE);
  }

  @Test
  public void testCallbackWithNoURLAsset() {
    Asset asset = ServiceData.newTestInstance();
    asset.setAddressable(null);
    callback.callback(asset, "123", Action.POST, ActionType.DEVICE);
  }

  @Test
  public void testCallbackWithAssets() {
    Asset asset1 = ServiceData.newTestInstance();
    Asset asset2 = ServiceData.newTestInstance();
    List<Asset> assets = new ArrayList<>();
    assets.add(asset1);
    assets.add(asset2);
    callback.callback(assets, "123", Action.POST, ActionType.DEVICE);
  }

  public class TestHandler implements HttpHandler {

    @Override

    public void handle(HttpExchange he) throws IOException {
      String response = "yep";
      he.sendResponseHeaders(200, response.length());
      OutputStream os = he.getResponseBody();
      os.write(response.getBytes());
      os.close();
    }
  }

}
