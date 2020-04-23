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

package org.edgexfoundry.controller.impl;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;

import org.edgexfoundry.controller.Action;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.domain.meta.CallbackAlert;
import org.edgexfoundry.exception.controller.ServiceException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

@Component
public class CallbackExecutor {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(CallbackExecutor.class);

  // TODO - someday build cache for callback URLs based on Addressable Id

  @Value("${server.timeout}")
  private int timeout = 5000;

  public void callback(List<Asset> assets, String id, Action action, ActionType type) {
    assets.forEach(a -> callback(a, id, action, type));
  }

  @Async
  public void callback(Asset asset, final String id, final Action action, final ActionType type) {
    if (asset == null) return;
    final String url = getCallBackURL(asset.getAddressable());
    try {
      if (url != null) {
        URL obj = new URL(url);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();
        con.setRequestMethod(action.toString());
        con.setDoOutput(true);
        con.setConnectTimeout(timeout);
        con.setRequestProperty("Content-Type", "application/json");
        String body;
        body = getBody(id, type);

        System.out.println(url + " - " + body);

        con.setRequestProperty("Content-Length", Integer.toString(body.length()));
        OutputStream os = con.getOutputStream();
        os.write(body.getBytes());

        int returnCode;
        returnCode = con.getResponseCode();
        logger.info("Call back device service @:  " + url + " with:  " + body
            + " received status code:  " + returnCode);
      } else
        logger.info("No address provided for " + action + " callback on :  " + getBody(id, type));
    } catch (Exception e) {
      logger.error("Trouble calling " + action.toString() + " callback on device service @:  " + url
          + " with id:  " + id + e.getMessage());
    }
  }

  private String getBody(String id, ActionType type) {
    CallbackAlert alert = new CallbackAlert(type, id);
    ObjectMapper mapper = new ObjectMapper();

    try {
      return mapper.writeValueAsString(alert);
    } catch (JsonProcessingException e) {
      logger.error(e.getMessage(), e);
      throw new ServiceException(e);
    }
  }

  private String getCallBackURL(Addressable addressable) {
    if (addressable != null) {
      StringBuilder builder = new StringBuilder(addressable.getProtocol().toString());
      builder.append("://");
      builder.append(addressable.getAddress());
      builder.append(":");
      builder.append(addressable.getPort());
      builder.append(addressable.getPath());
      return builder.toString();
    }
    return null;
  }
}
