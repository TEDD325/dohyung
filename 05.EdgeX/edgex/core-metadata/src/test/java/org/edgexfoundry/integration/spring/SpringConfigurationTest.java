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

package org.edgexfoundry.integration.spring;

import static org.junit.Assert.assertNotNull;

import org.edgexfoundry.Application;
import org.edgexfoundry.HeartBeat;
import org.edgexfoundry.dao.AddressableRepository;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class SpringConfigurationTest {


  @Autowired
  HeartBeat heartBeat;

  @Autowired
  ApplicationContext ctx;

  @Test
  public void testHeartBeatBeanExists() {
    assertNotNull("HeartBeat bean not available", heartBeat);
  }

  @Test
  @SuppressWarnings("rawtypes")
  public void testReposBeansExist() {
    Class[] clazz =
        {AddressableRepository.class, CommandRepository.class, DeviceProfileRepository.class,
            DeviceReportRepository.class, DeviceReportRepository.class, DeviceRepository.class,
            DeviceServiceRepository.class, ScheduleEventRepository.class, ScheduleRepository.class};
    checkBeanExistence(clazz);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private void checkBeanExistence(Class[] classes) {
    for (Class class1 : classes) {
      Object obj = ctx.getBean(class1);
      assertNotNull("Bean of type " + class1 + " was not found", obj);
    }
  }
}
